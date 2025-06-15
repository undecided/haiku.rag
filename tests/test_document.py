import pytest
from datasets import Dataset

from haiku.rag.store.engine import Store
from haiku.rag.store.models.document import Document


@pytest.mark.asyncio
async def test_create_document_with_chunks(qa_corpus: Dataset):
    """Test creating a document with chunks from the qa_corpus."""
    # Create an in-memory store
    store = Store(":memory:")
    
    # Get the first document from the corpus
    first_doc = qa_corpus[0]
    document_text = first_doc["document_extracted"]
    
    # Create a Document instance
    document = Document(
        content=document_text,
        metadata={"source": "qa_corpus", "topic": first_doc.get("document_topic", "")}
    )
    
    # Create the document with chunks in the database
    created_document = await document.create_with_chunks(store)
    
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
async def test_search_chunks(qa_corpus: Dataset):
    """Test vector search functionality."""
    # Create an in-memory store
    store = Store(":memory:")
    
    # Get the first document from the corpus
    first_doc = qa_corpus[0]
    document_text = first_doc["document_extracted"]
    
    # Create and store a document
    document = Document(
        content=document_text,
        metadata={"source": "qa_corpus"}
    )
    await document.create_with_chunks(store)
    
    # Perform a search
    search_query = "news"  # Simple query
    results = await Document.search_chunks(store, search_query, limit=3)
    
    # Verify search results
    assert len(results) <= 3
    assert all(hasattr(chunk, "content") for chunk in results)
    assert all(hasattr(chunk, "document_id") for chunk in results)
    assert all(chunk.document_id == document.id for chunk in results)
    
    store.close()