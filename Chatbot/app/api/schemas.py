from pydantic import BaseModel, Field

class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    session_id: str = Field(default="anon", max_length=64)