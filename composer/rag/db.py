from contextlib import asynccontextmanager
from typing import AsyncGenerator, cast, Any, LiteralString
import asyncio
import logging

import psycopg
from sentence_transformers import SentenceTransformer
from numpy import ndarray

from composer.rag.types import ManualRef, BlockChunk, ManualSectionHit
from composer.rag.text import code_ref_tag

logger = logging.getLogger(__name__)


DEFAULT_CONNECTION: str = "postgresql://rag_user:rag_password@localhost:5432/rag_db"

class PostgreSQLRAGDatabase:
    """Handle PostgreSQL database operations for RAG"""

    def __init__(self, conn_string: str, model: SentenceTransformer):
        self.conn_string = conn_string
        self.tr = model

    async def test_connection(self) -> None:
        """Test database connection and setup"""
        try:
            async with self._get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
                    logger.info("✅ Database connection successful")

                    # Check if documents table exists
                    await cur.execute("""
                        SELECT 1 FROM information_schema.tables
                        WHERE table_name = 'documents'
                    """)
                    if not await cur.fetchone():
                        logger.warning("❌ Documents table not found, creating...")
                        await self._create_schema()
                    else:
                        logger.info("✅ Documents table found")

        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    @asynccontextmanager
    async def _get_cursor(self) -> AsyncGenerator[psycopg.AsyncCursor[Any], None]:
        async with self._get_connection() as conn:
            async with conn.cursor() as cur:
                yield cur
            await conn.commit()

    @asynccontextmanager
    async def _get_connection(self) -> AsyncGenerator[psycopg.AsyncConnection[Any], None]:
        """Get database connection with context manager"""
        conn = None
        try:
            conn = await psycopg.AsyncConnection.connect(self.conn_string)
            yield conn
        except Exception as e:
            if conn:
                await conn.rollback()
            raise e
        finally:
            if conn:
                await conn.close()

    async def _create_schema(self) -> None:
        """Create database schema"""
        async with self._get_connection() as conn:
            async with conn.cursor() as cur:
                # create vector extension
                await cur.execute("""
                    CREATE EXTENSION IF NOT EXISTS vector;
                """)

                # Create documents table
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        content TEXT,
                        embedding vector(768),
                        h1 TEXT,
                        h2 TEXT,
                        h3 TEXT,
                        h4 TEXT,
                        h5 TEXT,
                        h6 TEXT,
                        created_at TIMESTAMP DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS code_refs (
                        id SERIAL PRIMARY KEY,
                        ref_number INTEGER,
                        code_body TEXT,
                        parent_doc integer REFERENCES documents(id)
                    );
                """)

                # Create indexes
                await cur.execute("""
                    CREATE INDEX IF NOT EXISTS documents_embedding_idx
                    ON documents USING hnsw (embedding vector_cosine_ops);
                """)

                await cur.execute("""
                    CREATE INDEX IF NOT EXISTS code_refs_lkp ON code_refs(parent_doc);
                """)

                await cur.execute("""
                    CREATE INDEX IF NOT EXISTS section_h1 ON documents (h1);
                    CREATE INDEX IF NOT EXISTS section_h2 ON documents (h2);
                    CREATE INDEX IF NOT EXISTS section_h3 ON documents (h3);
                    CREATE INDEX IF NOT EXISTS section_h4 ON documents (h4);
                """)

                await cur.execute("""
                    CREATE EXTENSION IF NOT EXISTS pg_trgm;
                    CREATE TABLE IF NOT EXISTS manual_sections(
                        id SERIAL PRIMARY KEY,
                        content TEXT,
                        h1 TEXT,
                        h2 TEXT,
                        h3 TEXT,
                        h4 TEXT,
                        h5 TEXT,
                        h6 TEXT,
                        part INTEGER,
                        created_at TIMESTAMP DEFAULT NOW(),
                        CONSTRAINT parts_unique UNIQUE (h1, h2, h3, h4, h5, h6, part)
                    );
                    CREATE INDEX IF NOT EXISTS manual_ts_idx ON manual_sections USING gin(
                        to_tsvector('english', content)
                    );
                    CREATE INDEX IF NOT EXISTS manual_trgm_idx ON manual_sections USING gin(
                        content gin_trgm_ops
                    );

                    CREATE TABLE IF NOT EXISTS manual_section_code_refs(
                        id INTEGER,
                        code_body TEXT,
                        section_id INTEGER,
                        CONSTRAINT id_section_id_pk PRIMARY KEY(id, section_id),
                        CONSTRAINT section_id_manual_section_fk FOREIGN KEY(section_id) REFERENCES manual_sections(id)
                    );
                """)

                await cur.execute("CREATE INDEX IF NOT EXISTS documents_content_idx ON documents USING gin(to_tsvector('english', content));")

                await conn.commit()
                logger.info("✅ Database schema created successfully")

    async def add_manual_section(self, ch: BlockChunk):
        async with self._get_cursor() as cur:
            headers : list[str | None] = [None] * 6
            for (ind, h) in enumerate(ch.headers):
                headers[ind] = h
            data = (ch.chunk,) + tuple(headers) + (ch.part,)
            await cur.execute("""
                INSERT INTO manual_sections(
                    content, h1, h2, h3, h4, h5, h6, part
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, data)
            insert_res = await cur.fetchone()
            if insert_res is None:
                raise Exception("Insertion didn't return ID")
            payloads = []
            for (i, code) in enumerate(ch.code_refs):
                payloads.append((i, code, insert_res[0]))
            await cur.executemany("""
                INSERT INTO manual_section_code_refs(
                    id, code_body, section_id
                ) VALUES (%s, %s, %s)
            """, payloads)


    async def add_chunks_batch(self, chunks: list[BlockChunk]) -> None:
        """Add chunks to database in batches"""
        if not chunks:
            return
        # SentenceTransformer.encode_document is CPU-bound (runs the embedding model);
        # offload to a thread to avoid blocking the event loop.
        embeddings = cast(list[ndarray], await asyncio.to_thread(
            self.tr.encode_document, [f"search_document: {d.chunk}" for d in chunks], show_progress_bar=False
        ))

        logger.info(f"Adding {len(chunks)} chunks to database...")
        # Insert batch
        async with self._get_connection() as conn:
            async with conn.cursor() as cur:
                for chunk, embedding in zip(chunks, embeddings):
                    try:
                        headers = tuple([f if f else None for f in chunk.headers])
                        payload = (chunk.chunk, embedding.tolist()) + headers
                        await cur.execute("""
                            INSERT INTO documents
                            (content, embedding, h1, h2, h3, h4, h5, h6)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            RETURNING id
                        """, payload)
                        insert_res = await cur.fetchone()
                        if insert_res is None:
                            raise Exception("Insertion didn't return any data")
                        new_id = insert_res[0]
                        for (i, code) in enumerate(chunk.code_refs):
                            await cur.execute("""
                                INSERT INTO code_refs (ref_number, code_body, parent_doc) VALUES (%s, %s, %s)
                            """, (
                                i, code, new_id
                            ))
                    except Exception as e:
                        logger.error(f"Failed to insert chunk {chunk}: {e}")
                        continue

                await conn.commit()

    async def find_refs(self, query: str, similarity_cutoff: float = 0.5, top_k: int = 10, manual_section : list[str] = []) -> list[ManualRef]:
        # SentenceTransformer.encode_query is CPU-bound (runs the embedding model);
        # offload to a thread to avoid blocking the event loop.
        question_embedding = cast(ndarray, await asyncio.to_thread(
            self.tr.encode_query, f"search_query: {query}", show_progress_bar=False
        ))

        params: tuple[Any, ...] = (question_embedding.tolist(),)
        where_clause = ""
        if len(manual_section) > 0:
            clauses = []
            for i in range(1, 7):
                params = params + (tuple(manual_section),)
                clauses.append(f"h{i} in %s")
            where_clause = "WHERE (" + " OR ".join(clauses) + ")"
        params += (question_embedding.tolist(), top_k)
        async with self._get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    SELECT id, content, 1 - (embedding <=> %s::vector) AS cosine_similarity, h1, h2, h3, h4, h5, h6
                    FROM documents
                    {where_clause}
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s
                """, params)

                res = await cur.fetchall()
                to_ret = []

                for row in res:
                    body : str = row[1]
                    similarity = row[2]
                    if similarity < similarity_cutoff:
                        break
                    header: list[str] = []
                    for i in row[3:]:
                        if i is None:
                            break
                        assert isinstance(i, str)
                        header.append(i)
                    await cur.execute(
                        """
                            SELECT ref_number, code_body FROM
                            code_refs WHERE parent_doc = %s
                        """, (row[0], )
                    )
                    async for code_row in cur:
                        id = code_row[0]
                        to_replace = code_ref_tag(id)
                        body = body.replace(to_replace, code_row[1])
                    to_ret.append(ManualRef(headers=header, content=body, similarity=similarity))
                return to_ret

    async def _replace_manual_code_refs(self, cur: psycopg.AsyncCursor[Any], content: str, section_id: int) -> str:
        await cur.execute(
            "SELECT id, code_body FROM manual_section_code_refs WHERE section_id = %s",
            (section_id,)
        )
        async for row in cur:
            content = content.replace(code_ref_tag(row[0]), row[1])
        return content

    async def search_manual_keywords(self, query: str, *, min_depth: int = 0, limit: int = 10) -> list[ManualSectionHit]:
        if min_depth < 0 or min_depth > 6:
            raise ValueError("min_depth must be between 0 and 6")
        depth_clause = f"AND h{min_depth} IS NOT NULL" if min_depth > 0 else ""
        depth_clause = cast(LiteralString, depth_clause)
        async with self._get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    SELECT ts_rank(to_tsvector('english', content), websearch_to_tsquery('english', %s)) AS relevance,
                           h1, h2, h3, h4, h5, h6
                    FROM manual_sections
                    WHERE to_tsvector('english', content) @@ websearch_to_tsquery('english', %s)
                    {depth_clause}
                    ORDER BY relevance DESC
                    LIMIT %s
                """, (query, query, limit))
                results = []
                for row in await cur.fetchall():
                    headers = [h for h in row[1:7] if h is not None]
                    results.append(ManualSectionHit(headers=headers, relevance=row[0]))
                return results

    async def get_manual_section(self, headers: list[str]) -> str | None:
        padded: list[str | None] = list(headers) + [None] * (6 - len(headers))
        padded = padded[:6]
        async with self._get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT id, content, part
                    FROM manual_sections
                    WHERE h1 IS NOT DISTINCT FROM %s
                      AND h2 IS NOT DISTINCT FROM %s
                      AND h3 IS NOT DISTINCT FROM %s
                      AND h4 IS NOT DISTINCT FROM %s
                      AND h5 IS NOT DISTINCT FROM %s
                      AND h6 IS NOT DISTINCT FROM %s
                    ORDER BY part ASC
                """, tuple(padded))
                rows = await cur.fetchall()
                if not rows:
                    return None
                parts = []
                for row in rows:
                    parts.append(await self._replace_manual_code_refs(cur, row[1], row[0]))
                return "\n".join(parts)
