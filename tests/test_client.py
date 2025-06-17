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
