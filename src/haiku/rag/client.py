import tempfile
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx

from haiku.rag.reader import FileReader
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

    async def create_document_from_source(
        self, source: str | Path, metadata: dict | None = None
    ) -> Document:
        """Create a document from a file path or URL.

        Args:
            source: File path (as string or Path) or URL to parse
            metadata: Optional metadata dictionary

        Returns:
            Created Document instance

        Raises:
            ValueError: If the file/URL cannot be parsed or doesn't exist
            httpx.RequestError: If URL request fails
        """

        # Check if it's a URL
        source_str = str(source)
        parsed_url = urlparse(source_str)
        if parsed_url.scheme in ("http", "https"):
            return await self._create_document_from_url(source_str, metadata)

        # Handle as file path
        source_path = Path(source) if isinstance(source, str) else source
        if source_path.suffix.lower() not in FileReader.extensions:
            raise ValueError(f"Unsupported file extension: {source_path.suffix}")

        if not source_path.exists():
            raise ValueError(f"File does not exist: {source_path}")

        content = FileReader.parse_file(source_path)

        # Create the document
        return await self.create_document(
            content=content, uri=str(source_path.resolve()), metadata=metadata
        )

    async def _create_document_from_url(
        self, url: str, metadata: dict | None = None
    ) -> Document:
        """Create a document from a URL by downloading and parsing the content.

        Args:
            url: URL to download and parse
            metadata: Optional metadata dictionary

        Returns:
            Created Document instance

        Raises:
            ValueError: If the content cannot be parsed
            httpx.RequestError: If URL request fails
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()

            # Get content type to determine file extension
            content_type = response.headers.get("content-type", "").lower()

            # Try to determine file extension from content type or URL
            file_extension = self._get_extension_from_content_type_or_url(
                url, content_type
            )

            if file_extension not in FileReader.extensions:
                raise ValueError(
                    f"Unsupported content type/extension: {content_type}/{file_extension}"
                )

            # Create a temporary file with the appropriate extension
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=file_extension, delete=False
            ) as temp_file:
                temp_file.write(response.content)
                temp_path = Path(temp_file.name)

            try:
                # Parse the content using FileReader
                content = FileReader.parse_file(temp_path)

                # Create the document with the original URL as URI
                return await self.create_document(
                    content=content, uri=url, metadata=metadata
                )
            finally:
                # Clean up temporary file
                temp_path.unlink(missing_ok=True)

    def _get_extension_from_content_type_or_url(
        self, url: str, content_type: str
    ) -> str:
        """Determine file extension from content type or URL."""
        # Common content type mappings
        content_type_map = {
            "text/html": ".html",
            "text/plain": ".txt",
            "text/markdown": ".md",
            "application/pdf": ".pdf",
            "application/json": ".json",
            "text/csv": ".csv",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        }

        # Try content type first
        for ct, ext in content_type_map.items():
            if ct in content_type:
                return ext

        # Try URL extension
        parsed_url = urlparse(url)
        path = Path(parsed_url.path)
        if path.suffix:
            return path.suffix.lower()

        # Default to .html for web content
        return ".html"

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
