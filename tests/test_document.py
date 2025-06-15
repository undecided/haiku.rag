import pytest
from datasets import Dataset

from haiku.rag.store.engine import Store
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.document import DocumentRepository


@pytest.mark.asyncio
async def test_create_document_with_chunks(qa_corpus: Dataset):
    """Test creating a document with chunks from the qa_corpus using repository."""
    # Create an in-memory store and repository
    store = Store(":memory:")
    doc_repo = DocumentRepository(store)
    
    # Get the first document from the corpus
    first_doc = qa_corpus[0]
    document_text = first_doc["document_extracted"]
    
    # Create a Document instance
    document = Document(
        content=document_text,
        metadata={"source": "qa_corpus", "topic": first_doc.get("document_topic", "")}
    )
    
    # Create the document with chunks in the database
    created_document = await doc_repo.create(document)
    
    # Verify the document was created
    assert created_document.id is not None
    assert created_document.content == document_text
    
    # Check that chunks were created in the database
    if store._connection is not None:
        cursor = store._connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (created_document.id,))
        chunk_count = cursor.fetchone()[0]
        
        assert chunk_count > 0
        
        # Check that embeddings were created
        cursor.execute("""
            SELECT COUNT(*) FROM chunk_embeddings ce
            JOIN chunks c ON c.id = ce.chunk_id
            WHERE c.document_id = ?
        """, (created_document.id,))
        embedding_count = cursor.fetchone()[0]
        
        assert embedding_count == chunk_count
        
        # Verify chunk metadata contains order information
        cursor.execute("SELECT metadata FROM chunks WHERE document_id = ? ORDER BY id", (created_document.id,))
        chunk_metadata = cursor.fetchall()
        
        for i, (metadata_json,) in enumerate(chunk_metadata):
            import json
            metadata = json.loads(metadata_json)
            assert "order" in metadata
            assert metadata["order"] == i
    
    store.close()


@pytest.mark.asyncio
async def test_document_repository_crud(qa_corpus: Dataset):
    """Test CRUD operations in DocumentRepository."""
    # Create an in-memory store and repository
    store = Store(":memory:")
    doc_repo = DocumentRepository(store)
    
    # Get the first document from the corpus
    first_doc = qa_corpus[0]
    document_text = first_doc["document_extracted"]
    
    # Create a document
    document = Document(
        content=document_text,
        metadata={"source": "test"}
    )
    created_document = await doc_repo.create(document)
    
    # Test get_by_id
    assert created_document.id is not None
    retrieved_document = await doc_repo.get_by_id(created_document.id)
    assert retrieved_document is not None
    assert retrieved_document.content == document_text
    
    # Test update (should regenerate chunks)
    retrieved_document.content = "Updated content for testing"
    updated_document = await doc_repo.update(retrieved_document)
    assert updated_document.content == "Updated content for testing"
    
    # Test list_all
    all_documents = await doc_repo.list_all()
    assert len(all_documents) == 1
    assert all_documents[0].id == created_document.id
    
    # Test delete
    deleted = await doc_repo.delete(created_document.id)
    assert deleted is True
    
    # Verify document is gone
    retrieved_document = await doc_repo.get_by_id(created_document.id)
    assert retrieved_document is None
    
    store.close()