from datetime import datetime, timezone
from typing import List, Optional, Type, Generic, TypeVar
from pydantic import BaseModel, Field

database = {}

class BaseDBModel(BaseModel):
    id: str = Field(default_factory=str)
    created_at: datetime = datetime.now(timezone.utc)
    last_updated: datetime = datetime.now(timezone.utc)

    @classmethod
    def query_by_id(cls, id: str):
        return database.get(cls.__name__.lower(), {}).get(id)
    
    def save_to_db(self) -> None:
        table_name = self.__class__.__name__.lower()
        if not table_name in database:
            database[table_name] = {}
        database[table_name][self.id] = self



