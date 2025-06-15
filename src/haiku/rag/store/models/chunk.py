from pydantic import BaseModel


class Chunk(BaseModel):
    """
    Represents a document with an ID, content, and metadata.
    """

    id: int | None = None
    document_id: int
    content: str
    metadata: dict = {}
