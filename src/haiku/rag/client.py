from pathlib import Path
from typing import Literal

from haiku.rag.store.engine import Store
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.document import DocumentRepository


class RAGClient:
    """High-level haiku-rag client."""

    def __init__(self, db_path: Path | Literal[":memory:"]):
        """Initialize the RAG client with a database path."""
        self.store = Store(db_path)
        self.document_repository = DocumentRepository(self.store)

    async def create_document(
        self, content: str, uri: str | None = None, metadata: dict | None = None
    ) -> Document:
        """Create a new document with optional URI and metadata."""
        document = Document(
            content=content,
            uri=uri,
            metadata=metadata or {},
        )
        return await self.document_repository.create(document)

    async def get_document_by_id(self, document_id: int) -> Document | None:
        """Get a document by its ID."""
        return await self.document_repository.get_by_id(document_id)

    async def get_document_by_uri(self, uri: str) -> Document | None:
        """Get a document by its URI."""
        return await self.document_repository.get_by_uri(uri)

    async def update_document(self, document: Document) -> Document:
        """Update an existing document."""
        return await self.document_repository.update(document)

    async def delete_document(self, document_id: int) -> bool:
        """Delete a document by its ID."""
        return await self.document_repository.delete(document_id)

    async def list_documents(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[Document]:
        """List all documents with optional pagination."""
        return await self.document_repository.list_all(limit=limit, offset=offset)

    def close(self):
        """Close the underlying store connection."""
        self.store.close()
