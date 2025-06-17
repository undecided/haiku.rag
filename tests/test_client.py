import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from datasets import Dataset

from haiku.rag.client import RAGClient


@pytest.mark.asyncio
async def test_client_document_crud(qa_corpus: Dataset):
    """Test RAGClient CRUD operations for documents."""
    # Create client with in-memory database
    client = RAGClient(":memory:")

    # Get test data
    first_doc = qa_corpus[0]
    document_text = first_doc["document_extracted"]
    test_uri = "file:///path/to/test.txt"
    test_metadata = {"source": "test", "topic": "testing"}

    # Test create_document
    created_doc = await client.create_document(
        content=document_text, uri=test_uri, metadata=test_metadata
    )

    assert created_doc.id is not None
    assert created_doc.content == document_text
    assert created_doc.uri == test_uri
    assert created_doc.metadata == test_metadata

    # Test get_document_by_id
    retrieved_doc = await client.get_document_by_id(created_doc.id)
    assert retrieved_doc is not None
    assert retrieved_doc.id == created_doc.id
    assert retrieved_doc.content == document_text
    assert retrieved_doc.uri == test_uri

    # Test get_document_by_uri
    retrieved_by_uri = await client.get_document_by_uri(test_uri)
    assert retrieved_by_uri is not None
    assert retrieved_by_uri.id == created_doc.id
    assert retrieved_by_uri.content == document_text

    # Test get_document_by_uri with non-existent URI
    non_existent = await client.get_document_by_uri("file:///non/existent.txt")
    assert non_existent is None

    # Test update_document
    retrieved_doc.content = "Updated content"
    retrieved_doc.uri = "file:///updated/path.txt"
    updated_doc = await client.update_document(retrieved_doc)
    assert updated_doc.content == "Updated content"
    assert updated_doc.uri == "file:///updated/path.txt"

    # Test list_documents
    all_docs = await client.list_documents()
    assert len(all_docs) == 1
    assert all_docs[0].id == created_doc.id

    # Test list_documents with pagination
    limited_docs = await client.list_documents(limit=10, offset=0)
    assert len(limited_docs) == 1

    # Test delete_document
    deleted = await client.delete_document(created_doc.id)
    assert deleted is True

    # Verify document is gone
    retrieved_doc = await client.get_document_by_id(created_doc.id)
    assert retrieved_doc is None

    # Test delete non-existent document
    deleted_again = await client.delete_document(created_doc.id)
    assert deleted_again is False

    client.close()


@pytest.mark.asyncio
async def test_client_create_document_from_source():
    """Test creating a document from a file source."""
    client = RAGClient(":memory:")

    # Create a temporary text file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        test_content = "This is test content from a file."
        f.write(test_content)
        temp_path = Path(f.name)

    try:
        # Test create_document_from_source with Path
        doc = await client.create_document_from_source(
            source=temp_path, metadata={"source_type": "file"}
        )

        assert doc.id is not None
        assert doc.content == test_content
        assert doc.uri == str(temp_path.resolve())
        assert doc.metadata["source_type"] == "file"

        # Test create_document_from_source with string path
        doc2 = await client.create_document_from_source(source=str(temp_path))

        assert doc2.id is not None
        assert doc2.content == test_content
        assert doc2.uri == str(temp_path.resolve())

    finally:
        # Clean up
        temp_path.unlink()
        client.close()


@pytest.mark.asyncio
async def test_client_create_document_from_source_unsupported():
    """Test creating a document from an unsupported file type."""
    client = RAGClient(":memory:")

    # Create a temporary file with unsupported extension
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".unsupported", delete=False
    ) as f:
        f.write("content")
        temp_path = Path(f.name)

    try:
        # Should raise ValueError for unsupported extension
        with pytest.raises(ValueError, match="Unsupported file extension"):
            await client.create_document_from_source(temp_path)

    finally:
        temp_path.unlink()
        client.close()


@pytest.mark.asyncio
async def test_client_create_document_from_source_nonexistent():
    """Test creating a document from a non-existent file."""
    client = RAGClient(":memory:")

    non_existent_path = Path("/non/existent/file.txt")

    # Should raise ValueError when file doesn't exist
    with pytest.raises(ValueError, match="File does not exist"):
        await client.create_document_from_source(non_existent_path)

    client.close()


@pytest.mark.asyncio
async def test_client_create_document_from_url():
    """Test creating a document from a URL."""
    client = RAGClient(":memory:")

    # Mock the HTTP response
    mock_response = AsyncMock()
    mock_response.content = b"<html><body><h1>Test Page</h1><p>This is test content from a webpage.</p></body></html>"
    mock_response.headers = {"content-type": "text/html"}
    mock_response.raise_for_status = AsyncMock()

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        doc = await client.create_document_from_source(
            source="https://example.com/test.html", metadata={"source_type": "web"}
        )

        assert doc.id is not None
        assert "Test Page" in doc.content
        assert "test content" in doc.content
        assert doc.uri == "https://example.com/test.html"
        assert doc.metadata["source_type"] == "web"

    client.close()


@pytest.mark.asyncio
async def test_client_create_document_from_url_with_different_content_types():
    """Test creating documents from URLs with different content types."""
    client = RAGClient(":memory:")

    # Test JSON content
    mock_json_response = AsyncMock()
    mock_json_response.content = (
        b'{"title": "Test JSON", "content": "This is JSON content"}'
    )
    mock_json_response.headers = {"content-type": "application/json"}
    mock_json_response.raise_for_status = AsyncMock()

    with patch("httpx.AsyncClient.get", return_value=mock_json_response):
        doc = await client.create_document_from_source(
            "https://api.example.com/data.json"
        )

        assert doc.id is not None
        assert "Test JSON" in doc.content
        assert doc.uri == "https://api.example.com/data.json"

    # Test plain text content
    mock_text_response = AsyncMock()
    mock_text_response.content = b"This is plain text content from a URL."
    mock_text_response.headers = {"content-type": "text/plain"}
    mock_text_response.raise_for_status = AsyncMock()

    with patch("httpx.AsyncClient.get", return_value=mock_text_response):
        doc = await client.create_document_from_source("https://example.com/readme.txt")

        assert doc.id is not None
        assert doc.content == "This is plain text content from a URL."
        assert doc.uri == "https://example.com/readme.txt"

    client.close()


@pytest.mark.asyncio
async def test_client_create_document_from_url_unsupported_content():
    """Test creating a document from URL with unsupported content type."""
    client = RAGClient(":memory:")

    # Mock response with unsupported content type
    mock_response = AsyncMock()
    mock_response.content = b"binary content"
    mock_response.headers = {"content-type": "application/octet-stream"}
    mock_response.raise_for_status = AsyncMock()

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        with pytest.raises(ValueError, match="Unsupported content type"):
            await client.create_document_from_source("https://example.com/binary.bin")

    client.close()


@pytest.mark.asyncio
async def test_client_create_document_from_url_http_error():
    """Test handling HTTP errors when creating document from URL."""
    client = RAGClient(":memory:")

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.side_effect = httpx.HTTPStatusError(
            "404 Not Found",
            request=httpx.Request("GET", "https://example.com/notfound.html"),
            response=httpx.Response(404),
        )

        with pytest.raises(httpx.HTTPStatusError):
            await client.create_document_from_source(
                "https://example.com/notfound.html"
            )

    client.close()


@pytest.mark.asyncio
async def test_get_extension_from_content_type_or_url():
    """Test the helper method for determining file extensions."""
    client = RAGClient(":memory:")

    # Test content type mappings
    assert client._get_extension_from_content_type_or_url("", "text/html") == ".html"
    assert (
        client._get_extension_from_content_type_or_url("", "application/pdf") == ".pdf"
    )
    assert client._get_extension_from_content_type_or_url("", "text/plain") == ".txt"

    # Test URL extension detection
    assert (
        client._get_extension_from_content_type_or_url(
            "https://example.com/doc.pdf", ""
        )
        == ".pdf"
    )
    assert (
        client._get_extension_from_content_type_or_url(
            "https://example.com/data.json", ""
        )
        == ".json"
    )

    # Test default fallback
    assert (
        client._get_extension_from_content_type_or_url("https://example.com/", "")
        == ".html"
    )

    # Test content type priority over URL extension
    assert (
        client._get_extension_from_content_type_or_url(
            "https://example.com/file.txt", "application/pdf"
        )
        == ".pdf"
    )

    client.close()
