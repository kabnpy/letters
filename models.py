from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class TimeMetrics(BaseModel):
    day_of_week: str
    hour_of_day: int
    is_weekend: bool
    travel_time: float | None = None
    response_delay_time: float | None = None


class TextMetrics(BaseModel):
    character_count: int
    word_count: int
    paragraph_count: int
    sentence_count: int
    average_word_length: float
    average_sentence_length: float
    sentences: list[str]


class EmotionalSignature(BaseModel):
    dominant_tone: str
    secondary_tone: str
    sentiment_score: float
    topics_discussed: list[str]
    memorable_internal_phrases: list[str]


class Letter(BaseModel):
    id: int
    sender: str
    direction: Literal["sent", "received"]
    body: str
    created_at: datetime
    deliver_at: datetime
    read_at: datetime | None = None

    # optional blocks representing downstream pipeline expansion
    temporal: TimeMetrics | None = None
    metrics: TextMetrics | None = None
    emotional_signature: EmotionalSignature | None = None
    relationship_stage: (
        Literal[
            "introduction",
            "building_rapport",
            "deep_vulnerability",
            "establishing_comfort",
        ]
        | None
    ) = None
    analysis_status: Literal["pending", "success", "failed"] = "pending"


class RelationshipArchive(BaseModel):
    generated_at: datetime
    total_letters: int
    letters: list[Letter]
