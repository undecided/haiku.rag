from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.models.document import Document


class HaikuRAGApp:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.console = Console()

    async def list_documents(self):
        async with HaikuRAG(db_path=self.db_path) as self.client:
            documents = await self.client.list_documents()
            for doc in documents:
                self._rich_print_document(doc, truncate=True)

    async def add_document_from_text(self, text: str):
        async with HaikuRAG(db_path=self.db_path) as self.client:
            doc = await self.client.create_document(text)
            self._rich_print_document(doc, truncate=True)
            self.console.print(
                f"[b]Document with id [cyan]{doc.id}[/cyan] added successfully.[/b]"
            )

    async def add_document_from_source(self, file_path: Path):
        async with HaikuRAG(db_path=self.db_path) as self.client:
            doc = await self.client.create_document_from_source(file_path)
            self._rich_print_document(doc, truncate=True)
            self.console.print(
                f"[b]Document with id [cyan]{doc.id}[/cyan] added successfully.[/b]"
            )

    async def get_document(self, doc_id: int):
        async with HaikuRAG(db_path=self.db_path) as self.client:
            doc = await self.client.get_document_by_id(doc_id)
            if doc is None:
                self.console.print(f"[red]Document with id {doc_id} not found.[/red]")
                return
            self._rich_print_document(doc, truncate=False)

    async def delete_document(self, doc_id: int):
        async with HaikuRAG(db_path=self.db_path) as self.client:
            await self.client.delete_document(doc_id)
            self.console.print(f"[b]Document {doc_id} deleted successfully.[/b]")

    async def search(self, query: str, limit: int = 5, k: int = 60):
        async with HaikuRAG(db_path=self.db_path) as self.client:
            results = await self.client.search(query, limit=limit, k=k)
            if not results:
                self.console.print("[red]No results found.[/red]")
                return
            for chunk, score in results:
                self._rich_print_search_result(chunk, score)

    def _rich_print_document(self, doc: Document, truncate: bool = False):
        """Format a document for display."""
        if truncate:
            content = doc.content.splitlines()
            if len(content) > 3:
                content = content[:3] + ["\n…"]
            content = "\n".join(content)
            content = Markdown(content)
        else:
            content = Markdown(doc.content)
        self.console.print(
            f"[repr.attrib_name]id[/repr.attrib_name]: {doc.id} [repr.attrib_name]uri[/repr.attrib_name]: {doc.uri} [repr.attrib_name]meta[/repr.attrib_name]: {doc.metadata}"
        )
        self.console.print(
            f"[repr.attrib_name]created at[/repr.attrib_name]: {doc.created_at} [repr.attrib_name]updated at[/repr.attrib_name]: {doc.updated_at}"
        )
        self.console.print("[repr.attrib_name]content[/repr.attrib_name]:")
        self.console.print(content)
        self.console.rule()

    def _rich_print_search_result(self, chunk: Chunk, score: float):
        """Format a search result chunk for display."""
        content = Markdown(chunk.content)
        self.console.print(
            f"[repr.attrib_name]document_id[/repr.attrib_name]: {chunk.document_id} "
            f"[repr.attrib_name]score[/repr.attrib_name]: {score:.4f}"
        )
        self.console.print("[repr.attrib_name]content[/repr.attrib_name]:")
        self.console.print(content)
        self.console.rule()

    def serve(self, transport: str | None = None):
        """Start the MCP server."""
        from haiku.rag.mcp import create_mcp_server

        server = create_mcp_server(self.db_path)

        if transport == "stdio":
            self.console.print("[green]Starting MCP server on stdio...[/green]")
            server.run("stdio")
        elif transport == "sse":
            self.console.print(
                "[green]Starting MCP server with streamable HTTP...[/green]"
            )
            server.run("sse")
        else:
            self.console.print("[green]Starting MCP server with HTTP...[/green]")
            server.run("streamable-http")
