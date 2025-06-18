# Haiku SQLite RAG

A SQLite-based Retrieval-Augmented Generation (RAG) system built for efficient document storage, chunking, and hybrid search capabilities.

## Features
- **Local SQLite**: No need to run additional servers
- **Support for various embedding providers**: You can use Ollama, VoyageAI, OpenAI or add your own
- **Vector Embeddings**: Uses sqlite-vec for efficient similarity search
- **Hybrid Search**: Full-text search (FTS5) combined with vector embeddings using Reciprocal Rank Fusion
- **Multi-format Support**: Parse 40+ file formats including PDF, DOCX, HTML, Markdown, audio and more
- **Web Content**: Direct URL ingestion with automatic content type detection

## Installation

```bash
uv pip install haiku.rag
```

By default Ollama (with the `mxbai-embed-large` model) is used for the embeddings.
For other providers use:

- **VoyageAI**: `uv pip install haiku.rag --extra voyageai`

## Configuration

If you want to use an alternative embeddings provider (Ollama being the default) you will need to set the provider details through environment variables:

By default:

```bash
EMBEDDING_PROVIDER="ollama"
EMBEDDING_MODEL="mxbai-embed-large" # or any other model
EMBEDDING_VECTOR_DIM=1024
```

For VoyageAI:
```bash
EMBEDDING_PROVIDER="voyageai"
EMBEDDING_MODEL="voyage-3.5" # or any other model
EMBEDDING_VECTOR_DIM=1024
```

## Quick Start

```python
from pathlib import Path
from haiku.rag.client import HaikuRAG

# Use as async context manager (recommended)
async with HaikuRAG("path/to/database.db") as client:
    # Create document from text
    doc = await client.create_document(
        content="Your document content here",
        uri="doc://example",
        metadata={"source": "manual", "topic": "example"}
    )

    # Create document from file (auto-parses content)
    doc = await client.create_document_from_source("path/to/document.pdf")

    # Create document from URL
    doc = await client.create_document_from_source("https://example.com/article.html")

    # Retrieve documents
    doc = await client.get_document_by_id(1)
    doc = await client.get_document_by_uri("file:///path/to/document.pdf")

    # List all documents with pagination
    docs = await client.list_documents(limit=10, offset=0)

    # Update document content
    doc.content = "Updated content"
    await client.update_document(doc)

    # Delete document
    await client.delete_document(doc.id)

    # Search documents using hybrid search (vector + full-text)
    results = await client.search("machine learning algorithms", limit=5)
    for chunk, score in results:
        print(f"Score: {score:.3f}")
        print(f"Content: {chunk.content}")
        print(f"Document ID: {chunk.document_id}")
        print("---")


# Or use without the context manager.
client = HaikuRAG(":memory:")
try:
    # ... operations ...
finally:
    client.close()
```

## Search Functionality

`haiku.rag` provides hybrid search combining vector similarity and full-text search:
1. **Vector Search**: Uses embeddings to find semantically similar content
2. **Full-text Search**: Uses SQLite FTS5 for exact keyword matching
3. **Hybrid Ranking**: Combines both using Reciprocal Rank Fusion (RRF)
4. **Chunked Results**: Returns relevant document chunks with scores

```python
async with HaikuRAG("database.db") as client:

    results = await client.search(
        query="machine learning",
        limit=5,  # Maximum results to return, defaults to 5
        k=60      # RRF parameter for reciprocal rank fusion, defaults to 60
    )

    # Process results
    for chunk, relevance_score in results:
        print(f"Relevance: {relevance_score:.3f}")
        print(f"Content: {chunk.content}")
        print(f"From document: {chunk.document_id}")
```


## Supported File Formats

`haiku.rag` supports 40+ file formats through MarkItDown:

- **Documents**: PDF, DOCX, PPTX, XLSX
- **Web**: HTML, XML
- **Text**: TXT, MD, CSV, JSON, YAML
- **Code**: PY, JS, TS, C, CPP, JAVA, GO, RS, and more
- **Media**: MP3, WAV (transcription)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest`
5. Run type checking & linting with `pyright` & `ruff check`
6. Submit a pull request
