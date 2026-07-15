import re
import sys
from datetime import datetime, timezone

from config import (
    ENRICHED_BASE_FILE,
    RAW_LETTERS_FILE,
)
from models import Letter, RelationshipArchive, TextMetrics, TimeMetrics


def calculate_text_metrics(text: str) -> TextMetrics:
    """Compute basic structural and linguistic statistics of a text block."""
    paragraphs = [p for p in text.split("\n") if p.strip()]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])", text) if s.strip()]
    sentence_count = max(len(sentences), 1)

    words = text.split()
    word_count = len(words)
    character_count = len(text)

    average_word_length = (
        round(sum(len(w) for w in words) / word_count, 2) if word_count else 0.0
    )
    average_sentence_length = round(word_count / sentence_count, 2)

    return TextMetrics(
        character_count=character_count,
        word_count=word_count,
        paragraph_count=len(paragraphs),
        sentence_count=sentence_count,
        average_word_length=average_word_length,
        average_sentence_length=average_sentence_length,
        sentences=sentences,
    )


def enrich_letters() -> None:
    """Read unprocessed letters, inject temporal and textual metrics, and save."""
    print(f"parsing {RAW_LETTERS_FILE}...")

    if not RAW_LETTERS_FILE.exists():
        raise FileNotFoundError(
            f"source file {RAW_LETTERS_FILE} missing. run fetch stage first"
        )

    with RAW_LETTERS_FILE.open("r", encoding="utf-8") as file:
        archive = RelationshipArchive.model_validate_json(file.read())

    # Letters arrive pre-sorted from the fetch stage; re-sort defensively.
    archive.letters.sort(key=lambda letter: letter.created_at)

    last_sent: Letter | None = None
    last_received: Letter | None = None

    for letter in archive.letters:
        travel_time = round(
            (letter.deliver_at - letter.created_at).total_seconds() / 3600, 2
        )
        response_delay = None
        if letter.direction == "sent" and last_received:
            delta = letter.created_at - last_received.deliver_at
            response_delay = round(max(delta.total_seconds() / 3600, 0.0), 2)
            last_received = None
        elif letter.direction == "received" and last_sent:
            delta = letter.created_at - last_sent.deliver_at
            response_delay = round(max(delta.total_seconds() / 3600, 0.0), 2)
            last_sent = None

        if letter.direction == "sent":
            last_sent = letter
        else:
            last_received = letter

        letter.temporal = TimeMetrics(
            day_of_week=letter.created_at.strftime("%A"),
            hour_of_day=letter.created_at.hour,
            is_weekend=letter.created_at.weekday() >= 5,
            travel_time=travel_time,
            response_delay_time=response_delay,
        )

        letter.metrics = calculate_text_metrics(letter.body)

    enriched_archive = RelationshipArchive(
        generated_at=datetime.now(timezone.utc),
        total_letters=len(archive.letters),
        letters=archive.letters,
    )

    with ENRICHED_BASE_FILE.open("w", encoding="utf-8") as file:
        file.write(enriched_archive.model_dump_json(indent=2))

    print(f"successfully enriched {enriched_archive.total_letters} letters into: {ENRICHED_BASE_FILE}")


if __name__ == "__main__":
    try:
        enrich_letters()
    except Exception as exc:
        print(f"enrich stage failed: {exc}")
        sys.exit(1)
