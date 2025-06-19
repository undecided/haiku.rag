# Haiku SQLite RAG

A SQLite-based Retrieval-Augmented Generation (RAG) system built for efficient document storage, chunking, and hybrid search capabilities.

## Features
- **Local SQLite**: No need to run additional servers
- **Support for various embedding providers**: You can use Ollama, VoyageAI or add your own
- **Hybrid Search**: Vector search using `sqlite-vec` combined with full-text search `FTS5`, using Reciprocal Rank Fusion
- **Multi-format Support**: Parse 40+ file formats including PDF, DOCX, HTML, Markdown, audio and more. Or add a url!

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

## Command Line Interface

`haiku.rag` includes a CLI application for managing documents and performing searches from the command line:

### Available Commands

```bash
# List all documents
haiku-rag list

# Add document from text
haiku-rag add "Your document content here"

# Add document from file or URL
haiku-rag add-src /path/to/document.pdf
haiku-rag add-src https://example.com/article.html

# Get and display a specific document
haiku-rag get 1

# Delete a document by ID
haiku-rag delete 1

# Search documents
haiku-rag search "machine learning"

# Search with custom options
haiku-rag search "python programming" --limit 10 --k 100

# Start MCP server (default HTTP transport)
haiku-rag serve # --stdio for stdio transport or --sse for SSE transport
```

All commands support the `--db` option to specify a custom database path. Run
```bash
haiku-rag command -h
```
to see additional parameters for a command.

## MCP Server

`haiku.rag` includes a Model Context Protocol (MCP) server that exposes RAG functionality as tools for AI assistants like Claude Desktop. The MCP server provides the following tools:

- `add_document_from_file` - Add documents from local file paths
- `add_document_from_url` - Add documents from URLs
- `add_document_from_text` - Add documents from raw text content
- `search_documents` - Search documents using hybrid search
- `get_document` - Retrieve specific documents by ID
- `list_documents` - List all documents with pagination
- `delete_document` - Delete documents by ID

You can start the server (using Streamble HTTP, stdio or SSE transports) with:

```bash
# Start with default HTTP transport
haiku-rag serve # --stdio for stdio transport or --sse for SSE transport
```

## Using `haiku.rag` from python

### Managing documents

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
```

## Searching documents

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
