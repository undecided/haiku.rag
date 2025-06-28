import asyncio
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.mcp import create_mcp_server
from haiku.rag.monitor import FileWatcher
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

    async def ask(self, question: str):
        async with HaikuRAG(db_path=self.db_path) as self.client:
            try:
                answer = await self.client.ask(question)
                self.console.print(f"[bold blue]Question:[/bold blue] {question}")
                self.console.print()
                self.console.print("[bold green]Answer:[/bold green]")
                self.console.print(Markdown(answer))
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")

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

    async def serve(self, transport: str | None = None):
        """Start the MCP server."""
        async with HaikuRAG(self.db_path) as client:
            monitor = FileWatcher(paths=Config.MONITOR_DIRECTORIES, client=client)
            monitor_task = asyncio.create_task(monitor.observe())
            server = create_mcp_server(self.db_path)

            try:
                if transport == "stdio":
                    await server.run_stdio_async()
                elif transport == "sse":
                    await server.run_sse_async("sse")
                else:
                    await server.run_http_async("streamable-http")
            except KeyboardInterrupt:
                pass
            finally:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
