import json

from haiku.rag.chunker import chunker
from haiku.rag.embeddings.ollama import Embedder
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.repositories.base import BaseRepository


class ChunkRepository(BaseRepository[Chunk]):
    """Repository for Chunk database operations."""

    def __init__(self, store, embedder: Embedder | None = None):
        super().__init__(store)
        self.embedder = embedder or Embedder()

    async def create(self, entity: Chunk, commit: bool = True) -> Chunk:
        """Create a chunk in the database."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.cursor()
        cursor.execute(
            """
            INSERT INTO chunks (document_id, content, metadata)
            VALUES (?, ?, ?)
            """,
            (entity.document_id, entity.content, json.dumps(entity.metadata)),
        )

        entity.id = cursor.lastrowid

        # Generate and store embedding
        embedding = await self.embedder.embed(entity.content)
        serialized_embedding = self.store.serialize_embedding(embedding)
        cursor.execute(
            """
            INSERT INTO chunk_embeddings (chunk_id, embedding)
            VALUES (?, ?)
            """,
            (entity.id, serialized_embedding),
        )

        if commit:
            self.store._connection.commit()
        return entity

    async def get_by_id(self, entity_id: int) -> Chunk | None:
        """Get a chunk by its ID."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.cursor()
        cursor.execute(
            """
            SELECT id, document_id, content, metadata
            FROM chunks WHERE id = ?
            """,
            (entity_id,),
        )

        row = cursor.fetchone()
        if row is None:
            return None

        chunk_id, document_id, content, metadata_json = row
        metadata = json.loads(metadata_json) if metadata_json else {}

        return Chunk(
            id=chunk_id, document_id=document_id, content=content, metadata=metadata
        )

    async def update(self, entity: Chunk) -> Chunk:
        """Update an existing chunk."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")
        if entity.id is None:
            raise ValueError("Chunk ID is required for update")

        cursor = self.store._connection.cursor()
        cursor.execute(
            """
            UPDATE chunks
            SET document_id = ?, content = ?, metadata = ?
            WHERE id = ?
            """,
            (
                entity.document_id,
                entity.content,
                json.dumps(entity.metadata),
                entity.id,
            ),
        )

        # Regenerate and update embedding
        embedding = await self.embedder.embed(entity.content)
        serialized_embedding = self.store.serialize_embedding(embedding)
        cursor.execute(
            """
            UPDATE chunk_embeddings
            SET embedding = ?
            WHERE chunk_id = ?
            """,
            (serialized_embedding, entity.id),
        )

        self.store._connection.commit()
        return entity

    async def delete(self, entity_id: int, commit: bool = True) -> bool:
        """Delete a chunk by its ID."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.cursor()

        # Delete the embedding first
        cursor.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (entity_id,))

        # Delete the chunk
        cursor.execute("DELETE FROM chunks WHERE id = ?", (entity_id,))

        deleted = cursor.rowcount > 0
        if commit:
            self.store._connection.commit()
        return deleted

    async def list_all(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[Chunk]:
        """List all chunks with optional pagination."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.cursor()
        query = "SELECT id, document_id, content, metadata FROM chunks ORDER BY document_id, id"
        params = []

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        if offset is not None:
            query += " OFFSET ?"
            params.append(offset)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        chunks = []
        for row in rows:
            chunk_id, document_id, content, metadata_json = row
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

    async def create_chunks_for_document(
        self, document_id: int, content: str, commit: bool = True
    ) -> list[Chunk]:
        """Create chunks and embeddings for a document."""
        # Chunk the document content
        chunk_texts = await chunker.chunk(content)
        created_chunks = []

        # Create chunks with embeddings using the create method
        for order, chunk_text in enumerate(chunk_texts):
            # Create chunk with order in metadata
            chunk = Chunk(
                document_id=document_id, content=chunk_text, metadata={"order": order}
            )

            created_chunk = await self.create(chunk, commit=commit)
            created_chunks.append(created_chunk)

        return created_chunks

    async def delete_by_document_id(
        self, document_id: int, commit: bool = True
    ) -> bool:
        """Delete all chunks for a document."""
        chunks = await self.get_by_document_id(document_id)

        deleted_any = False
        for chunk in chunks:
            if chunk.id is not None:
                deleted = await self.delete(chunk.id, commit=False)
                deleted_any = deleted_any or deleted

        if commit and deleted_any and self.store._connection:
            self.store._connection.commit()
        return deleted_any

    async def search_chunks(self, query: str, limit: int = 5) -> list[Chunk]:
        """Search for relevant chunks using vector similarity."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.cursor()

        # Generate embedding for the query
        query_embedding = await self.embedder.embed(query)
        serialized_query_embedding = self.store.serialize_embedding(query_embedding)

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
            chunk_id, document_id, content, metadata_json, _ = row
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

    async def get_by_document_id(self, document_id: int) -> list[Chunk]:
        """Get all chunks for a specific document."""
        if self.store._connection is None:
            raise ValueError("Store connection is not available")

        cursor = self.store._connection.cursor()
        cursor.execute(
            """
            SELECT id, document_id, content, metadata
            FROM chunks WHERE document_id = ?
            ORDER BY JSON_EXTRACT(metadata, '$.order')
            """,
            (document_id,),
        )

        rows = cursor.fetchall()
        chunks = []

        for row in rows:
            chunk_id, document_id, content, metadata_json = row
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
