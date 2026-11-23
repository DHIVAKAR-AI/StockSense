from pydantic import BaseModel


class AssistantQueryRequest(BaseModel):
    message: str
