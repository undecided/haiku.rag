# Python API

Use `haiku.rag` directly in your Python applications.

## Basic Usage

```python
from pathlib import Path
from haiku.rag.client import HaikuRAG

# Use as async context manager (recommended)
async with HaikuRAG("path/to/database.db") as client:
    # Your code here
    pass
```

## Document Management

### Creating Documents

From text:
```python
doc = await client.create_document(
    content="Your document content here",
    uri="doc://example",
    metadata={"source": "manual", "topic": "example"}
)
```

From file:
```python
doc = await client.create_document_from_source("path/to/document.pdf")
```

From URL:
```python
doc = await client.create_document_from_source("https://example.com/article.html")
```

### Retrieving Documents

By ID:
```python
doc = await client.get_document_by_id(1)
```

By URI:
```python
doc = await client.get_document_by_uri("file:///path/to/document.pdf")
```

List all documents:
```python
docs = await client.list_documents(limit=10, offset=0)
```

### Updating Documents

```python
doc.content = "Updated content"
await client.update_document(doc)
```

### Deleting Documents

```python
await client.delete_document(doc.id)
```

## Searching Documents

Basic search:
```python
results = await client.search("machine learning algorithms", limit=5)
for chunk, score in results:
    print(f"Score: {score:.3f}")
    print(f"Content: {chunk.content}")
    print(f"Document ID: {chunk.document_id}")
```

With options:
```python
results = await client.search(
    query="machine learning",
    limit=5,  # Maximum results to return
    k=60      # RRF parameter for reciprocal rank fusion
)

# Process results
for chunk, relevance_score in results:
    print(f"Relevance: {relevance_score:.3f}")
    print(f"Content: {chunk.content}")
    print(f"From document: {chunk.document_id}")
```

## Search Technology

`haiku.rag` uses hybrid search combining:
- **Vector search** using `sqlite-vec` for semantic similarity
- **Full-text search** using SQLite's `FTS5` for keyword matching
- **Reciprocal Rank Fusion** to combine and rank results
