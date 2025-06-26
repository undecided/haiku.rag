# Configuration

Configuration is done through environment variables.

## File Monitoring

Set directories to monitor for automatic indexing:

```bash
# Monitor single directory
MONITOR_DIRECTORIES="/path/to/documents"

# Monitor multiple directories
MONITOR_DIRECTORIES="/path/to/documents,/another_path/to/documents"
```

## Embedding Providers

If you use Ollama, you can use any pulled model that supports embeddings.

### Ollama (Default)

```bash
EMBEDDINGS_PROVIDER="ollama"
EMBEDDINGS_MODEL="mxbai-embed-large"
EMBEDDINGS_VECTOR_DIM=1024
```

### VoyageAI
If you want to use VoyageAI embeddings you will need to install `haiku.rag` with the VoyageAI extras,

```bash
uv pip install haiku.rag --extra voyageai
```

```bash
EMBEDDINGS_PROVIDER="voyageai"
EMBEDDINGS_MODEL="voyage-3.5"
EMBEDDINGS_VECTOR_DIM=1024
VOYAGE_API_KEY="your-api-key"
```

### OpenAI
If you want to use OpenAI embeddings you will need to install `haiku.rag` with the VoyageAI extras,

```bash
uv pip install haiku.rag --extra openai
```

and set environment variables.

```bash
EMBEDDINGS_PROVIDER="openai"
EMBEDDINGS_MODEL="text-embedding-3-small"  # or text-embedding-3-large
EMBEDDINGS_VECTOR_DIM=1536
OPENAI_API_KEY="your-api-key"
```
