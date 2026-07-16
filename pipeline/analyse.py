# pipeline/analyse.py
import json
import logging
import sys
import time
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, OpenAI
from pydantic import ValidationError

from config import (
    ACTIVE_BACKEND,
    ENRICHED_BASE_FILE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SEMANTIC_ANALYSIS_FILE,
)
from models import EmotionalSignature, Letter, RelationshipArchive

logger = logging.getLogger(__name__)

# how many immediately-preceding, already analysed letters to summarise as
# tone continuity context.
CONTEXT_WINDOW = 3


@dataclass(frozen=True)
class BackendConfig:
    max_retries: int
    request_timeout: float


BACKEND_CONFIGS: dict[str, BackendConfig] = {
    "local": BackendConfig(
        max_retries=1,
        request_timeout=400.0,  # observed up to ~250s per letter on CPU
    ),
    "cloud": BackendConfig(
        max_retries=3,
        request_timeout=120.0,
    ),
}


def build_context_digest(archive_letters: list[Letter], current_index: int) -> str:
    """
    Summarise up to CONTEXT_WINDOW immediately-preceding letters that
    already have an emotional_signature, for tone-continuity purposes.

    This looks backward through the full chronological archive (not just
    this run's pending letters), so it picks up letters analysed in a
    previous run just as well as ones analysed earlier in this same run.
    Only letters with a signature already present are used — pending or
    permanently-failed neighbors contribute nothing, since there's no tone
    information to draw on yet.

    Returns an empty string if there's no prior context available (e.g.
    the very first letter, or a fresh archive with a gap right before this
    letter), in which case the prompt below falls back to letter-only
    analysis exactly as before.

    """
    recent: list[Letter] = []
    i = current_index - 1
    while i >= 0 and len(recent) < CONTEXT_WINDOW:
        neighbour = archive_letters[i]
        if neighbour.emotional_signature is not None:
            recent.append(neighbour)
        i -= 1
    recent.reverse()

    if not recent:
        return ""

    lines = [
        f"- {n.sender} wrote with a {n.emotional_signature.dominant_tone} "
        f"(secondary: {n.emotional_signature.secondary_tone}) tone, "
        f"sentiment {n.emotional_signature.sentiment_score:+.2f}, "
        f"discussing: {', '.join(n.emotional_signature.topics_discussed) or 'n/a'}"
        for n in recent
    ]
    return (
        "for continuity, here is the tone of the most recent preceding "
        "letter(s) in this correspondence (oldest first):\n" + "\n".join(lines)
    )


def _is_retryable(exc: Exception) -> bool:
    """
    Decide whether *exc* is worth retrying, based on its concrete type
    rather than duck-typing a status code off whatever attributes happen
    to be present.

    - ValidationError: the model responded, but not with valid
      EmotionalSignature JSON. Retrying with the identical prompt rarely
      helps, so this is treated as permanent.
    - APIConnectionError (includes APITimeoutError): connection-level
      issues — network blips, server not reachable, request timed out.
      Usually transient, worth retrying.
    - APIStatusError: the server responded with a 4xx/5xx. 429 (rate
      limited) and 5xx (server-side fault) are worth retrying; anything
      else (400 bad request, 401/403 auth, 404, etc.) means the request
      itself is wrong and won't succeed on a retry.
    """
    if isinstance(exc, ValidationError):
        return False
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


def analyse_single_letter(
    letter: Letter,
    context_digest: str,
    client: OpenAI,
    backend_cfg: BackendConfig,
) -> EmotionalSignature | None:
    """Send a letter to the configured LLM backend and return a parsed
    EmotionalSignature, or None if every attempt failed.

    Only exceptions from the openai client (APIConnectionError,
    APIStatusError, and subclasses like RateLimitError) and pydantic
    schema-validation failures are caught and treated as a per-letter
    failure — see _is_retryable for which of those are retried. Anything
    else (a genuine bug in this code, e.g. an AttributeError from a schema
    mismatch we didn't anticipate) propagates immediately instead of being
    silently recorded as "letter N failed", so real bugs surface as bugs.
    """

    schema = json.dumps(EmotionalSignature.model_json_schema(), indent=2)

    system_prompt = (
        "you are an expert interpersonal linguistic analyst tracking an archive of correspondence.\n"
        f"extract the emotional profile written by {letter.sender}.\n"
        "your response must be a single JSON object that conforms precisely to this JSON Schema:\n"
        f"{schema}"
    )
    user_prompt = f"extract the emotional profile and key internal phrases from this letter:\n\n{letter.body}"

    if context_digest:
        user_prompt = f"{context_digest}\n\nnow, {user_prompt}"

    for attempt in range(1, backend_cfg.max_retries + 1):
        logger.info(
            f"processing letter {letter.id} (sender: {letter.sender}, attempt {attempt}/{backend_cfg.max_retries})..."
        )
        start_time = time.time()

        try:
            response = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                timeout=backend_cfg.request_timeout,
            )

            raw_json_str = response.choices[0].message.content or ""
            result = EmotionalSignature.model_validate_json(raw_json_str)
            elapsed = time.time() - start_time
            logger.info(f"letter {letter.id} analyzed successfully in {elapsed:.1f}s")
            return result

        except (APIConnectionError, APIStatusError, ValidationError) as exc:
            elapsed = time.time() - start_time
            retryable = _is_retryable(exc)

            if not retryable or attempt == backend_cfg.max_retries:
                logger.error(
                    f"failed processing letter {letter.id} after {elapsed:.1f}s "
                    f"(attempt {attempt}/{backend_cfg.max_retries}, retryable={retryable}): {exc}"
                )
                return None

            wait = 2 ** (attempt - 1) * 3
            logger.warning(
                f"letter {letter.id} failed after {elapsed:.1f}s (attempt {attempt}/{backend_cfg.max_retries}), retrying in {wait}s: {exc}"
            )
            time.sleep(wait)

    return None  # unreachable, but keeps type-checkers happy


def save_checkpoint(archive: RelationshipArchive) -> None:
    """Atomically dump the current archive state to disk."""
    tmp_file = SEMANTIC_ANALYSIS_FILE.with_suffix(".tmp")
    try:
        with tmp_file.open("w", encoding="utf-8") as f:
            f.write(archive.model_dump_json(indent=2))
        tmp_file.replace(SEMANTIC_ANALYSIS_FILE)
    except OSError:
        logger.exception("failed to save progress checkpoint")


def load_archive_with_progress() -> tuple[RelationshipArchive, list[Letter]]:
    """Load the source archive, merge in any existing checkpointed progress,
    and return (archive, pending_letters).
    """
    if not ENRICHED_BASE_FILE.exists():
        raise FileNotFoundError(
            f"source data file '{ENRICHED_BASE_FILE}' not found. run stage 1 first!"
        )

    with ENRICHED_BASE_FILE.open("r", encoding="utf-8") as f:
        archive = RelationshipArchive.model_validate_json(f.read())

    # Keyed by letter id -> (signature, status) as recorded in the last
    # checkpoint. We carry both fields forward rather than re-deriving
    # status from "is signature present", so a letter that failed every
    # retry on a prior run stays visibly "failed" (not indistinguishable
    # from "never attempted") even though it remains eligible for retry.
    progress_map: dict[int, tuple[EmotionalSignature | None, str]] = {}
    if SEMANTIC_ANALYSIS_FILE.exists():
        try:
            with SEMANTIC_ANALYSIS_FILE.open("r", encoding="utf-8") as f:
                existing_archive = RelationshipArchive.model_validate_json(f.read())
                for letter in existing_archive.letters:
                    if (
                        letter.emotional_signature
                        or letter.analysis_status != "pending"
                    ):
                        progress_map[letter.id] = (
                            letter.emotional_signature,
                            letter.analysis_status,
                        )
            succeeded = sum(1 for sig, _ in progress_map.values() if sig is not None)
            failed = len(progress_map) - succeeded
            logger.info(
                f"loaded existing progress: {succeeded} letters already processed, "
                f"{failed} previously failed and will be retried."
            )
        except ValidationError as exc:
            logger.warning(
                f"could not read valid progress from checkpoint file: {exc}. "
                "fresh start initiated."
            )

    for letter in archive.letters:
        if letter.id in progress_map:
            signature, status = progress_map[letter.id]
            letter.emotional_signature = signature
            letter.analysis_status = status
        # Defensive invariant, independent of what was stored: a signature
        # being present always means "success", regardless of a stale or
        # missing status field from an older checkpoint format.
        if letter.emotional_signature is not None:
            letter.analysis_status = "success"

    pending = [
        letter for letter in archive.letters if letter.analysis_status != "success"
    ]
    return archive, pending


def analyse_semantics() -> None:
    """
    Orchestrate the semantic analysis pipeline.

    Each letter's prompt includes a short tone-continuity digest built
    from the most recent  *already-analysed* neighbors (see build_context_digest),
    so a letter can only be processed once its immediate predecessors have real
    signatures. Concurrent/out-of-order completion would break that
    guarantee. Letters are walked in the archive's existing chronological
    order (set by enrich.py), so a fresh run naturally builds up context
    letter-by-letter, and a resumed run picks up in the same order it left
    off in.

    Every resolved letter — success or permanent failure — triggers an
    immediate checkpoint write with an explicit analysis_status, so an
    interruption (Ctrl-C, crash, etc.) loses at most the one letter
    in flight, and the checkpoint always distinguishes "succeeded",
    "failed and awaiting retry", and "never attempted" for every letter.
    """
    if ACTIVE_BACKEND not in BACKEND_CONFIGS:
        raise ValueError(
            f"unknown backend {ACTIVE_BACKEND!r}. "
            f"set ACTIVE_BACKEND to one of: {list(BACKEND_CONFIGS)}"
        )

    backend_cfg = BACKEND_CONFIGS[ACTIVE_BACKEND]

    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", max_retries=0)

    archive, pending = load_archive_with_progress()
    total = len(archive.letters)

    if not pending:
        logger.info("all letters are already processed and up to date.")
        return

    logger.info(
        f"starting llm analytics pipeline ({ACTIVE_BACKEND} backend, sequential): "
        f"{len(pending)} remaining out of {total} total letters."
    )

    pending_ids = {letter.id for letter in pending}
    completed = 0
    failed = 0

    for index, letter in enumerate(archive.letters):
        if letter.id not in pending_ids:
            continue

        context_digest = build_context_digest(archive.letters, index)
        signature = analyse_single_letter(letter, context_digest, client, backend_cfg)

        if signature:
            letter.emotional_signature = signature
            letter.analysis_status = "success"
            save_checkpoint(archive)
            completed += 1
            logger.info(
                f"checkpoint saved (letter {letter.id}, {completed + failed}/{len(pending)} resolved this run, {completed} succeeded)."
            )
        else:
            letter.analysis_status = "failed"
            save_checkpoint(archive)
            failed += 1
            logger.error(
                f"letter {letter.id} failed permanently this run ({completed + failed}/{len(pending)} resolved, {failed} failed) — will retry on next run."
            )

    logger.info(
        f"pipeline execution completed. saved results to: '{SEMANTIC_ANALYSIS_FILE}'"
    )


if __name__ == "__main__":
    try:
        analyse_semantics()
    except Exception:
        logger.exception("analyse stage failed")
        sys.exit(1)
