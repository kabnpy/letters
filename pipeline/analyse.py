# pipeline/analyse.py
import asyncio
import json
import sys
import time

# ---------------------------------------------------------------------------
# Backend tuning
# ---------------------------------------------------------------------------
# These three knobs are the only things that should change when you switch
# between a local model server and a cloud provider. They're explicit on
# purpose rather than inferred from the URL, so you always know what's
# active without having to think through implicit logic.
#
#   concurrency  — how many requests are allowed to be in flight at once.
#                  Local CPU inference does not benefit from concurrency >1:
#                  there's no rate limit to dodge, and parallel requests just
#                  compete for the same cores, often making both slower.
#                  Cloud providers can genuinely run requests in parallel, so
#                  a higher number can meaningfully speed up a full run —
#                  start conservative and raise it only after confirming you
#                  don't see 429s in practice.
#
#   max_retries  — how many attempts a single letter gets before being
#                  marked as failed for this run (it will be retried on the
#                  next run via the checkpoint/resume mechanism regardless).
#                  Locally, failures are rarely transient (crashed server,
#                  OOM, bad timeout) so retrying repeatedly just delays
#                  seeing the real error. Remotely, failures are often
#                  genuinely transient (rate limits, brief network blips)
#                  and benefit from a few attempts with backoff.
#
#   request_timeout — per-request timeout in seconds. Set this above your
#                  observed worst-case latency, not a guess. Local CPU
#                  inference on long letters has been observed to take up
#                  to ~250s; cloud providers are typically far faster but
#                  can have cold starts.
#
# Set ACTIVE_BACKEND=cloud in your .env (or environment) to switch.
from dataclasses import dataclass

from openai import AsyncOpenAI
from pydantic import ValidationError

from config import (
    ACTIVE_BACKEND,
    ENRICHED_BASE_FILE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SEMANTIC_ANALYSIS_FILE,
)
from models import EmotionalSignature, Letter, RelationshipArchive

@dataclass(frozen=True)
class BackendConfig:
    concurrency: int
    max_retries: int
    request_timeout: float


BACKEND_CONFIGS: dict[str, BackendConfig] = {
    "local": BackendConfig(
        concurrency=1,
        max_retries=1,
        request_timeout=400.0,  # observed up to ~250s per letter on CPU
    ),
    "cloud": BackendConfig(
        concurrency=4,
        max_retries=3,
        request_timeout=120.0,
    ),
}


async def analyse_single_letter(
    letter: Letter,
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    backend_cfg: BackendConfig,
    total_count: int,
) -> tuple[int, EmotionalSignature | None]:
    """Send a letter to the configured LLM backend and return a parsed
    EmotionalSignature, or None if every attempt failed.

    Retries are only attempted for failures that look transient (connection
    errors, 429 rate limiting, 5xx server errors). Anything else (bad
    request, auth failure, validation failure against our schema) fails
    immediately rather than burning through retry attempts pointlessly.
    """
    schema = json.dumps(EmotionalSignature.model_json_schema(), indent=2)

    system_prompt = (
        "you are an expert interpersonal linguistic analyst tracking an archive of correspondence.\n"
        f"extract the emotional profile written by {letter.sender}.\n"
        "your response must be a single JSON object that conforms precisely to this JSON Schema:\n"
        f"{schema}"
    )
    user_prompt = f"extract the emotional profile and key internal phrases from this letter:\n\n{letter.body}"

    async with semaphore:
        for attempt in range(1, backend_cfg.max_retries + 1):
            print(
                f"processing letter {letter.id} (sender: {letter.sender}, attempt {attempt}/{backend_cfg.max_retries}) — {total_count} letters queued this run..."
            )
            start_time = time.time()

            try:
                response = await client.chat.completions.create(
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
                print(f"letter {letter.id} analyzed successfully in {elapsed:.1f}s")
                return letter.id, result

            except Exception as exc:
                elapsed = time.time() - start_time
                status = getattr(getattr(exc, "response", None), "status_code", None)
                # None status (e.g. connection/timeout errors) and 429/5xx
                # are treated as transient and worth retrying. Anything else
                # (400, 401, 403, schema validation failures) is treated as
                # permanent for this letter.
                retryable = (
                    status is None
                    or status == 429
                    or (status is not None and status >= 500)
                )

                if not retryable or attempt == backend_cfg.max_retries:
                    print(
                        f"failed processing letter {letter.id} after {elapsed:.1f}s "
                        f"(attempt {attempt}/{backend_cfg.max_retries}, retryable={retryable}): {exc}"
                    )
                    return letter.id, None

                wait = 2 ** (attempt - 1) * 3
                print(
                    f"letter {letter.id} failed after {elapsed:.1f}s (attempt {attempt}/{backend_cfg.max_retries}), retrying in {wait}s: {exc}"
                )
                await asyncio.sleep(wait)

    return letter.id, None  # unreachable, but keeps type-checkers happy


def save_checkpoint(archive: RelationshipArchive) -> None:
    """Atomically dump the current archive state to disk."""
    tmp_file = SEMANTIC_ANALYSIS_FILE.with_suffix(".tmp")
    try:
        with tmp_file.open("w", encoding="utf-8") as f:
            f.write(archive.model_dump_json(indent=2))
        tmp_file.replace(SEMANTIC_ANALYSIS_FILE)
    except Exception as exc:
        print(f"failed to save progress checkpoint: {exc}")


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

    progress_map: dict[int, EmotionalSignature] = {}
    if SEMANTIC_ANALYSIS_FILE.exists():
        try:
            with SEMANTIC_ANALYSIS_FILE.open("r", encoding="utf-8") as f:
                existing_archive = RelationshipArchive.model_validate_json(f.read())
                for letter in existing_archive.letters:
                    if letter.emotional_signature:
                        progress_map[letter.id] = letter.emotional_signature
            print(f"loaded existing progress: {len(progress_map)} letters already processed.")
        except ValidationError as exc:
            print(
                f"could not read valid progress from checkpoint file: {exc}. "
                "fresh start initiated."
            )

    for letter in archive.letters:
        if letter.id in progress_map:
            letter.emotional_signature = progress_map[letter.id]

    pending = [
        letter for letter in archive.letters if letter.emotional_signature is None
    ]
    return archive, pending


async def analyse_semantics_async() -> None:
    """Orchestrate the semantic analysis pipeline.

    Letters are launched as a pool of tasks bounded by concurrency via a
    semaphore. Results are consumed as they complete (not in submission
    order) via asyncio.as_completed, and EVERY successful letter triggers
    an immediate checkpoint write. This means an interruption (Ctrl-C,
    crash, etc.) loses at most the letters that were actively in flight at
    that moment — never an entire batch of completed-but-unsaved work.
    """
    if ACTIVE_BACKEND not in BACKEND_CONFIGS:
        raise ValueError(
            f"unknown backend {ACTIVE_BACKEND!r}. "
            f"set ACTIVE_BACKEND to one of: {list(BACKEND_CONFIGS)}"
        )

    backend_cfg = BACKEND_CONFIGS[ACTIVE_BACKEND]

    # max_retries=0 on the client itself: we handle retries explicitly above,
    # at the granularity of a single letter, with visible logging at each
    # attempt. Leaving the client's own silent retry behavior enabled would
    # double up retries invisibly and make timeouts harder to diagnose.
    client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", max_retries=0)

    archive, pending = load_archive_with_progress()
    total = len(archive.letters)

    if not pending:
        print("all letters are already processed and up to date.")
        return

    print(
        f"starting llm analytics pipeline ({ACTIVE_BACKEND} backend, concurrency={backend_cfg.concurrency}): "
        f"{len(pending)} remaining out of {total} total letters."
    )

    semaphore = asyncio.Semaphore(backend_cfg.concurrency)
    letters_by_id = {letter.id: letter for letter in archive.letters}

    tasks = [
        asyncio.create_task(
            analyse_single_letter(letter, semaphore, client, backend_cfg, len(pending))
        )
        for letter in pending
    ]

    completed = 0
    failed = 0
    for finished in asyncio.as_completed(tasks):
        letter_id, signature = await finished
        if signature:
            letters_by_id[letter_id].emotional_signature = signature
            save_checkpoint(archive)
            completed += 1
            print(
                f"checkpoint saved (letter {letter_id}, {completed + failed}/{len(pending)} resolved this run, {completed} succeeded)."
            )
        else:
            failed += 1
            print(
                f"letter {letter_id} failed permanently this run ({completed + failed}/{len(pending)} resolved, {failed} failed)."
            )

    print(f"pipeline execution completed. saved results to: '{SEMANTIC_ANALYSIS_FILE}'")


def analyse_semantics() -> None:
    """Synchronous entry point — wraps the async orchestrator."""
    asyncio.run(analyse_semantics_async())


if __name__ == "__main__":
    try:
        analyse_semantics()
    except Exception as exc:
        print(f"analyse stage failed: {exc}")
        sys.exit(1)
