
import asyncio
import os
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

from app.utils.ids import generate_uuid

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


class VectorStore:

    def __init__(self):
        self.create_table()

    def create_table(self):

        # Enable extension before registering the vector type
        with psycopg.connect(DATABASE_URL) as connection:

            connection.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_hash TEXT UNIQUE NOT NULL,
                    file_size_bytes BIGINT,
                    uploaded_at TEXT,
                    ingested_at TEXT,
                    ingestion_status TEXT,
                    chunk_count INTEGER,
                    chunking_version TEXT,
                    embedding_model TEXT
                );
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    filename_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    chunk_index INTEGER,
                    page_number INTEGER,
                    text TEXT NOT NULL,
                    embedding_model_name TEXT,
                    embedding VECTOR
                );
            """)

        print("Documents and chunks tables created successfully!")

    def get_connection(self):

        connection = psycopg.connect(
            DATABASE_URL,
            row_factory=dict_row
        )

        register_vector(connection)
        return connection

    async def get_document_by_hash(self, file_hash: str) -> dict | None:

        def _get():

            with self.get_connection() as connection:

                result = connection.execute("""
                    SELECT *
                    FROM documents
                    WHERE file_hash = %s
                    LIMIT 1;
                """, (file_hash,)).fetchone()

                return result

        return await asyncio.to_thread(_get)

    async def get_document_by_id(self, document_id: str) -> dict | None:

        def _get():

            with self.get_connection() as connection:

                return connection.execute("""
                    SELECT *
                    FROM documents
                    WHERE document_id = %s;
                """, (document_id,)).fetchone()

        return await asyncio.to_thread(_get)

    async def insert(
        self,
        document_record: dict,
        chunks: list[dict],
        embeddings: list[list[float]]
    ) -> None:

        def _insert():

            if len(chunks) != len(embeddings):
                raise ValueError("Chunks and embeddings count mismatch")

            with self.get_connection() as connection:

                # 1. Insert document metadata
                connection.execute("""
                    INSERT INTO documents (
                        document_id,
                        filename,
                        file_hash,
                        file_size_bytes,
                        uploaded_at,
                        ingested_at,
                        ingestion_status,
                        chunk_count,
                        chunking_version,
                        embedding_model
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """, (
                    document_record["document_id"],
                    document_record["filename"],
                    document_record["file_hash"],
                    document_record["file_size_bytes"],
                    document_record["uploaded_at"],
                    document_record["ingested_at"],
                    document_record["ingestion_status"],
                    document_record["chunk_count"],
                    document_record["chunking_version"],
                    document_record["embedding_model"]
                ))

                # 2. Insert document chunks
                for chunk, vector in zip(chunks, embeddings):

                    connection.execute("""
                        INSERT INTO chunks (
                            id,
                            filename_id,
                            filename,
                            chunk_index,
                            page_number,
                            text,
                            embedding_model_name,
                            embedding
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """, (
                        chunk.get("id") or generate_uuid(),
                        document_record["document_id"],
                        document_record["filename"],
                        int(chunk["chunk_index"]),
                        int(chunk["page_number"]),
                        chunk["text"],
                        document_record["embedding_model"],
                        vector
                    ))

                print(
                    f"Successfully stored {len(chunks)} chunks "
                    f"for {document_record['filename']}!"
                )

        await asyncio.to_thread(_insert)

    async def retrieve(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        document_id: str | None = None
    ) -> list[dict]:

        def _retrieve():

            with self.get_connection() as connection:

                if document_id:

                    results = connection.execute("""
                        SELECT *,
                               embedding <=> %s AS distance
                        FROM chunks
                        WHERE filename_id = %s
                        ORDER BY embedding <=> %s
                        LIMIT %s;
                    """, (
                        query_embedding,
                        document_id,
                        query_embedding,
                        n_results
                    )).fetchall()

                else:

                    results = connection.execute("""
                        SELECT *,
                               embedding <=> %s AS distance
                        FROM chunks
                        ORDER BY embedding <=> %s
                        LIMIT %s;
                    """, (
                        query_embedding,
                        query_embedding,
                        n_results
                    )).fetchall()

                return [
                    {
                        "id": row["id"],
                        "document": row["text"],
                        "metadata": {
                            "document_id": row["filename_id"],
                            "filename": row["filename"],
                            "chunk_index": row["chunk_index"],
                            "page_number": row["page_number"],
                            "embedding_model_name": row["embedding_model_name"]
                        },
                        "distance": row["distance"]
                    }
                    for row in results
                ]

        return await asyncio.to_thread(_retrieve)


vector_store = VectorStore()
