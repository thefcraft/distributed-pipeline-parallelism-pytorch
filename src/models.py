from pydantic import BaseModel, HttpUrl
from typing import Literal
from uuid import UUID

class BaseHeader(BaseModel):
    batch_key: UUID
    shape: tuple[int, ...]
    dtype: str
class ForwardHeader(BaseHeader):
    grad_req: bool = False

class ServerConfig(BaseModel):
    forward_server: HttpUrl
    backward_server: HttpUrl | None = None


class SuccessJobResponse(BaseModel):
    batch_key: UUID
    success: Literal[True] = True