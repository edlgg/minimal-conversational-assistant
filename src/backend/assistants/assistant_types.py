from datetime import datetime, timezone
from typing import List
from pydantic import BaseModel, Field
from backend.db import BaseDBModel


class Message(BaseModel):
    sent_by: str # assistant, correspondent
    text: str
    llm_name: str
    llm_model: str
    llm_temperature: float
    created_at: datetime = datetime.now(timezone.utc)

class AssistantState(BaseDBModel):
    messages: List[Message] = Field(default_factory=list)
