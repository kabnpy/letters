import sys
import time
from datetime import datetime, timezone

import requests
from pydantic import BaseModel, ValidationError

from config import RAW_LETTERS_FILE, SLOWLY_POST_ID, SLOWLY_TOKEN, SLOWLY_USER_ID
from models import Letter, RelationshipArchive


class _RawLetter(BaseModel):
    """Letter payload returned by the Slowly API."""

    id: int
    user: int
    user_to: int
    body: str
    name: str
    created_at: datetime
    deliver_at: datetime
    read_at: datetime | None = None


class _LetterPage(BaseModel):
    """Paginated collection of letters."""

    current_page: int
    data: list[_RawLetter]
    next_page_url: str | None = None


class _ApiResponse(BaseModel):
    """Top-level API response."""

    comments: _LetterPage
    now: str


def fetch_letters() -> list[Letter]:
    """Paginate through the Slowly API and return all letters as typed Letter objects."""
    letters: list[Letter] = []
    url = f"https://api.getslowly.com/friend/{SLOWLY_POST_ID}/all"
    current_page = 1

    print(f"fetching letters for post {SLOWLY_POST_ID}...")

    while url:
        try:
            response = requests.get(
                url,
                params={"token": SLOWLY_TOKEN},
                headers={"Accept": "application/json"},
                timeout=15,
            )
            response.raise_for_status()

            payload = _ApiResponse.model_validate(response.json())
            page_letters = payload.comments.data

            if not page_letters:
                break

            for raw in page_letters:
                letters.append(
                    Letter(
                        id=raw.id,
                        sender=raw.name,
                        direction="sent" if raw.user == SLOWLY_USER_ID else "received",
                        body=raw.body,
                        created_at=raw.created_at,
                        deliver_at=raw.deliver_at,
                        read_at=raw.read_at,
                    )
                )

            current_page = payload.comments.current_page
            print(
                f"page {current_page}: fetched {len(page_letters)} letters (total: {len(letters)})"
            )

            url = payload.comments.next_page_url
            if not url:
                break

            time.sleep(0.4)

        except (requests.RequestException, ValidationError) as exc:
            raise RuntimeError(f"error processing page {current_page}: {exc}") from exc

    letters.sort(key=lambda letter: letter.created_at)
    return letters


def write_export(letters: list[Letter]) -> None:
    """Serialize *letters* as a RelationshipArchive and write to RAW_LETTERS_FILE."""
    archive = RelationshipArchive(
        generated_at=datetime.now(timezone.utc),
        total_letters=len(letters),
        letters=letters,
    )

    with RAW_LETTERS_FILE.open("w", encoding="utf-8") as file:
        file.write(archive.model_dump_json(indent=2))

    print(f"exported {len(letters)} letters to {RAW_LETTERS_FILE}")


def fetch_letters_stage() -> None:
    """Entry point for the fetch stage of the pipeline."""
    letters = fetch_letters()
    write_export(letters)


if __name__ == "__main__":
    try:
        fetch_letters_stage()
    except Exception as exc:
        print(f"fetch stage failed: {exc}")
        sys.exit(1)
