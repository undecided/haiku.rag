# Haiku SQLite RAG

A SQLite-based Retrieval-Augmented Generation (RAG) system built for efficient document storage, chunking, and hybrid search capabilities.

## Features

- **Document Management**: Store and manage documents with automatic content parsing
- **Smart Updates**: Intelligent file/URL monitoring with MD5-based change detection
- **Hybrid Search**: Full-text search (FTS5) combined with vector embeddings
- **Multi-format Support**: Parse 40+ file formats including PDF, DOCX, HTML, Markdown, and more
- **Web Content**: Direct URL ingestion with automatic content type detection
- **Vector Embeddings**: Uses sqlite-vec for efficient similarity search
- **Automatic Chunking**: Intelligent document segmentation for better retrieval

## Installation

```bash
uv pip install haiku.rag
```

or for development, checkout the repository and then,

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

## Quick Start

```python
from pathlib import Path
from haiku.rag.client import HaikuRAG

# Initialize client with database path
client = HaikuRAG("path/to/database.db")
# Or use in-memory database for testing
client = HaikuRAG(":memory:")

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

# Clean up
client.close()
```

## Search Functionality

`haiku.rag` provides hybrid search combining vector similarity and full-text search:
1. **Vector Search**: Uses embeddings to find semantically similar content
2. **Full-text Search**: Uses SQLite FTS5 for exact keyword matching
3. **Hybrid Ranking**: Combines both using Reciprocal Rank Fusion (RRF)
4. **Chunked Results**: Returns relevant document chunks with scores

```python
# Basic search
results = await client.search("your query here")

# Search with custom parameters
results = await client.search(
    query="machine learning",
    limit=10,  # Maximum results to return
    k=60       # RRF parameter for reciprocal rank fusion
)

# Process results
for chunk, relevance_score in results:
    print(f"Relevance: {relevance_score:.3f}")
    print(f"Content: {chunk.content}")
    print(f"From document: {chunk.document_id}")
```

## Smart Document Updates

The system automatically tracks file changes using MD5 hashes:

```python
# First call - creates new document
doc1 = await client.create_document_from_source("document.txt")

# Second call - no changes, returns existing document (no processing)
doc2 = await client.create_document_from_source("document.txt")
assert doc1.id == doc2.id

# After file modification - automatically updates existing document
# File content changed...
doc3 = await client.create_document_from_source("document.txt")
assert doc1.id == doc3.id  # Same document
assert doc3.content != doc1.content  # Updated content
```

## Supported File Formats

The system supports 40+ file formats through MarkItDown:

- **Documents**: PDF, DOCX, PPTX, XLSX
- **Web**: HTML, XML
- **Text**: TXT, MD, CSV, JSON, YAML
- **Code**: PY, JS, TS, C, CPP, JAVA, GO, RS, and more
- **Media**: MP3, WAV (transcription)

## Document Metadata

Documents automatically include metadata:

```python
doc = await client.create_document_from_source("example.pdf")
print(doc.metadata)
# {
#   "contentType": "application/pdf",
#   "md5": "abc123...",
#   "custom_field": "value"  # Your custom metadata
# }
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest`
5. Run type checking & linting with `pyright` & `ruff check`
6. Submit a pull request
