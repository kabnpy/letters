import json
import logging
import sys
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    PROGRESSION_FILE,
    SEMANTIC_ANALYSIS_FILE,
)
from models import Letter, RelationshipArchive

logger = logging.getLogger(__name__)


class StageClassification(BaseModel):
    """Pydantic model for the LLM's classification response."""

    relationship_stage: Literal[
        "introduction", "building_rapport", "establishing_comfort", "deep_vulnerability"
    ]
    rationale: str


def build_progression_context(archive_letters: list[Letter], current_index: int) -> str:
    """Provide context of the relationship stage of the preceding letter."""
    if current_index == 0:
        return "This is the very first letter in the correspondence."

    prev_letter = archive_letters[current_index - 1]
    prev_stage = prev_letter.relationship_stage or "unknown"
    return (
        f"The immediately preceding letter was classified in the '{prev_stage}' stage."
    )


def classify_letter_stage_llm(
    letter: Letter,
    context_str: str,
    client: OpenAI,
) -> StageClassification | None:
    """Call the LLM to classify the relationship stage of the letter."""
    schema = json.dumps(StageClassification.model_json_schema(), indent=2)

    system_prompt = (
        "You are an expert relationship counselor and interpersonal analyst tracking correspondence.\n"
        "Analyze the content and tone of the letter, and classify the current stage of the pen-pal relationship.\n"
        "Choose exactly one of these stages:\n"
        "- 'introduction': Initial polite exchange, greetings, basic facts (hobbies, job, location).\n"
        "- 'building_rapport': Finding common ground, sharing daily life, storytelling, showing curiosity about each other.\n"
        "- 'establishing_comfort': Sharing deeper thoughts, personal opinions, values, and routines; showing emotional warmth and support.\n"
        "- 'deep_vulnerability': Discussing personal struggles, fears, family issues, deep secrets, or significant life transitions; high emotional intimacy.\n\n"
        "Your response must be a single JSON object conforming precisely to this JSON Schema:\n"
        f"{schema}"
    )

    user_prompt = (
        f"{context_str}\n\n"
        f"Sender: {letter.sender}\n"
        f"Direction: {letter.direction}\n"
        f"Tones: {letter.emotional_signature.dominant_tone if letter.emotional_signature else 'unknown'}\n"
        f"Topics: {', '.join(letter.emotional_signature.topics_discussed) if letter.emotional_signature else 'unknown'}\n"
        f"Letter Body:\n{letter.body}"
    )

    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            timeout=120.0,
        )
        raw_json = response.choices[0].message.content or ""
        return StageClassification.model_validate_json(raw_json)
    except Exception as exc:
        logger.error(f"Failed LLM classification for letter {letter.id}: {exc}")
        return None


def save_checkpoint(archive: RelationshipArchive) -> None:
    """Atomically save the progression progress to PROGRESSION_FILE."""
    tmp_file = PROGRESSION_FILE.with_suffix(".tmp")
    try:
        with tmp_file.open("w", encoding="utf-8") as f:
            f.write(archive.model_dump_json(indent=2))
        tmp_file.replace(PROGRESSION_FILE)
    except OSError:
        logger.exception("Failed to save progress checkpoint in progression stage")


def load_archive_with_progression() -> tuple[RelationshipArchive, list[Letter]]:
    """Load the semantic analysis output, merge existing progression progress, and return pending letters."""
    # Source file is SEMANTIC_ANALYSIS_FILE
    if not SEMANTIC_ANALYSIS_FILE.exists():
        raise FileNotFoundError(
            f"Source file {SEMANTIC_ANALYSIS_FILE} missing. Run semantic analysis (analyse) first."
        )

    with SEMANTIC_ANALYSIS_FILE.open("r", encoding="utf-8") as f:
        archive = RelationshipArchive.model_validate_json(f.read())

    # Sort chronologically
    archive.letters.sort(key=lambda letter: letter.created_at)

    # Merge progression checkpoints if PROGRESSION_FILE exists
    progress_map = {}
    if PROGRESSION_FILE.exists():
        try:
            with PROGRESSION_FILE.open("r", encoding="utf-8") as f:
                existing_archive = RelationshipArchive.model_validate_json(f.read())
                for letter in existing_archive.letters:
                    if letter.relationship_stage is not None:
                        progress_map[letter.id] = letter.relationship_stage
            logger.info(f"Loaded existing progression for {len(progress_map)} letters.")
        except Exception as exc:
            logger.warning(
                f"Could not load valid progression checkpoint: {exc}. Starting fresh."
            )

    for letter in archive.letters:
        if letter.id in progress_map:
            letter.relationship_stage = progress_map[letter.id]

    pending = [letter for letter in archive.letters if letter.relationship_stage is None]
    return archive, pending


def track_relationship_progression() -> None:
    """Orchestrate the progression tracking stage."""
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", max_retries=0)

    archive, pending = load_archive_with_progression()
    total = len(archive.letters)

    if not pending:
        logger.info("All letters already classified for relationship progression.")
        return

    logger.info(
        f"Starting relationship progression LLM classifier: "
        f"{len(pending)} remaining out of {total} total letters."
    )

    pending_ids = {letter.id for letter in pending}
    completed = 0
    failed = 0

    for index, letter in enumerate(archive.letters):
        if letter.id not in pending_ids:
            continue

        context_str = build_progression_context(archive.letters, index)
        classification = classify_letter_stage_llm(letter, context_str, client)

        if classification:
            letter.relationship_stage = classification.relationship_stage
            save_checkpoint(archive)
            completed += 1
            logger.info(
                f"Classified letter {letter.id} as '{classification.relationship_stage}' "
                f"(Rationale: {classification.rationale[:60]}...)"
            )
        else:
            failed += 1
            logger.error(f"Letter {letter.id} failed progression classification.")

    logger.info(
        f"Progression tracking completed. Results written to: {PROGRESSION_FILE}"
    )

    # Log/print transitions summary
    transitions = []
    last_stage = None
    first_date = archive.letters[0].created_at if archive.letters else None

    for idx, letter in enumerate(archive.letters):
        stage = letter.relationship_stage
        if stage and stage != last_stage:
            days = (letter.created_at - first_date).days if first_date else 0
            transitions.append(
                f"• {letter.created_at.strftime('%Y-%m-%d')} (Day {days}): "
                f"Transitioned to '{stage.upper()}' at letter index {idx}"
            )
            last_stage = stage

    print("\n--- Relationship Stage Transitions (LLM-based) ---")
    if transitions:
        print("\n".join(transitions))
    else:
        print("No stage transitions detected.")
    print("---------------------------------------------------\n")


if __name__ == "__main__":
    try:
        track_relationship_progression()
    except Exception:
        logger.exception("Progression stage failed")
        sys.exit(1)
