import asyncio
from pathlib import Path

import typer

from haiku.rag.app import HaikuRAGApp
from haiku.rag.utils import get_default_data_dir

cli = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]}, no_args_is_help=True
)

event_loop = asyncio.get_event_loop()


@cli.command("list", help="List all stored documents")
def list_documents(
    db: Path = typer.Option(
        get_default_data_dir() / "haiku.rag.sqlite",
        "--db",
        help="The path to the sqlite db to use",
    ),
):
    app = HaikuRAGApp(db_path=db)
    event_loop.run_until_complete(app.list_documents())


@cli.command("add", help="Add a document from text input")
def add_document_text(
    text: str = typer.Argument(
        help="The text content of the document to add",
    ),
    db: Path = typer.Option(
        get_default_data_dir() / "haiku.rag.sqlite",
        "--db",
        help="The path to the sqlite db to use",
    ),
):
    app = HaikuRAGApp(db_path=db)
    event_loop.run_until_complete(app.add_document_from_text(text=text))


@cli.command("add-src", help="Add a document from a file path or URL")
def add_document_src(
    file_path: Path = typer.Argument(
        help="The file path or URL of the document to add",
    ),
    db: Path = typer.Option(
        get_default_data_dir() / "haiku.rag.sqlite",
        "--db",
        help="The path to the sqlite db to use",
    ),
):
    app = HaikuRAGApp(db_path=db)
    event_loop.run_until_complete(app.add_document_from_source(file_path=file_path))


@cli.command("get", help="Get and display a document by its ID")
def get_document(
    doc_id: int = typer.Argument(
        help="The ID of the document to get",
    ),
    db: Path = typer.Option(
        get_default_data_dir() / "haiku.rag.sqlite",
        "--db",
        help="The path to the sqlite db to use",
    ),
):
    app = HaikuRAGApp(db_path=db)
    event_loop.run_until_complete(app.get_document(doc_id=doc_id))


@cli.command("delete", help="Delete a document by its ID")
def delete_document(
    doc_id: int = typer.Argument(
        help="The ID of the document to delete",
    ),
    db: Path = typer.Option(
        get_default_data_dir() / "haiku.rag.sqlite",
        "--db",
        help="The path to the sqlite db to use",
    ),
):
    app = HaikuRAGApp(db_path=db)
    event_loop.run_until_complete(app.delete_document(doc_id=doc_id))


@cli.command("search", help="Search for documents by a query")
def search(
    query: str = typer.Argument(
        help="The search query to use",
    ),
    limit: int = typer.Option(
        5,
        "--limit",
        "-l",
        help="Maximum number of results to return",
    ),
    k: int = typer.Option(
        60,
        "--k",
        help="Reciprocal Rank Fusion k parameter",
    ),
    db: Path = typer.Option(
        get_default_data_dir() / "haiku.rag.sqlite",
        "--db",
        help="The path to the sqlite db to use",
    ),
):
    app = HaikuRAGApp(db_path=db)
    event_loop.run_until_complete(app.search(query=query, limit=limit, k=k))


if __name__ == "__main__":
    cli()
