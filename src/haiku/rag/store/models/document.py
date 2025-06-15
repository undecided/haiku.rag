import json
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from haiku.rag.chunker import chunker
from haiku.rag.embeddings.ollama import Embedder
from haiku.rag.store.models.chunk import Chunk

if TYPE_CHECKING:
    from haiku.rag.store.engine import Store


class Document(BaseModel):
    """
    Represents a document with an ID, content, and metadata.
    """

    id: int | None = None
    content: str
    metadata: dict = {}
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    async def create_with_chunks(self, store: "Store") -> "Document":
        """
        Create a document in the database along with its chunks and embeddings.

        Args:
            store: The Store instance to use for database operations

        Returns:
            Document: The created document with updated id
        """
        if store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = store._connection.cursor()
        embedder = Embedder()

        # Insert the document
        cursor.execute(
            """
            INSERT INTO documents (content, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                self.content,
                json.dumps(self.metadata),
                self.created_at,
                self.updated_at,
            ),
        )

        document_id = cursor.lastrowid
        assert document_id is not None, "Failed to create document in database"
        self.id = document_id

        # Chunk the document content
        chunk_texts = await chunker.chunk(self.content)

        # Create chunks with embeddings
        for order, chunk_text in enumerate(chunk_texts):
            # Create chunk with order in metadata
            chunk = Chunk(
                document_id=document_id, content=chunk_text, metadata={"order": order}
            )

            cursor.execute(
                """
                INSERT INTO chunks (document_id, content, metadata)
                VALUES (?, ?, ?)
                """,
                (chunk.document_id, chunk.content, json.dumps(chunk.metadata)),
            )
            chunk_id = cursor.lastrowid

            # Generate and store embedding
            embedding = await embedder.embed(chunk_text)
            serialized_embedding = store.serialize_embedding(embedding)
            cursor.execute(
                """
                INSERT INTO chunk_embeddings (chunk_id, embedding)
                VALUES (?, ?)
                """,
                (chunk_id, serialized_embedding),
            )

        store._connection.commit()
        return self

    @classmethod
    async def search_chunks(
        cls, store: "Store", query: str, limit: int = 5
    ) -> list[Chunk]:
        """
        Search for relevant chunks using vector similarity with sqlite-vec.

        Args:
            store: The Store instance to use for database operations
            query: The text query to search for
            limit: Maximum number of chunks to return

        Returns:
            List of relevant Chunk objects ordered by similarity
        """
        if store._connection is None:
            raise ValueError("Store connection is not available")

        embedder = Embedder()
        cursor = store._connection.cursor()

        # Generate embedding for the query
        query_embedding = await embedder.embed(query)
        serialized_query_embedding = store.serialize_embedding(query_embedding)

        # Search for similar chunks using sqlite-vec
        cursor.execute(
            """
            SELECT c.id, c.document_id, c.content, c.metadata, distance
            FROM chunk_embeddings
            JOIN chunks c ON c.id = chunk_embeddings.chunk_id
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (serialized_query_embedding, limit),
        )

        results = cursor.fetchall()
        chunks = []

        for row in results:
            chunk_id, document_id, content, metadata_json, distance = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    content=content,
                    metadata=metadata,
                )
            )

        return chunks
