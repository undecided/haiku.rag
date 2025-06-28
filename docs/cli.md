# Command Line Interface

The `haiku-rag` CLI provides complete document management functionality.

## Document Management

### List Documents

```bash
haiku-rag list
```

### Add Documents

From text:
```bash
haiku-rag add "Your document content here"
```

From file or URL:
```bash
haiku-rag add-src /path/to/document.pdf
haiku-rag add-src https://example.com/article.html
```

### Get Document

```bash
haiku-rag get 1
```

### Delete Document

```bash
haiku-rag delete 1
```

## Search

Basic search:
```bash
haiku-rag search "machine learning"
```

With options:
```bash
haiku-rag search "python programming" --limit 10 --k 100
```

## Question Answering

Ask questions about your documents:
```bash
haiku-rag ask "Who is the author of haiku.rag?"
```

The QA agent will search your documents for relevant information and provide a comprehensive answer.

## Server

Start the MCP server:
```bash
# HTTP transport (default)
haiku-rag serve

# stdio transport
haiku-rag serve --stdio

# SSE transport
haiku-rag serve --sse
```

## Options

All commands support:
- `--db` - Specify custom database path
- `-h` - Show help for specific command

Example:
```bash
haiku-rag list --db /path/to/custom.db
haiku-rag add -h
```
