from contextlib import asynccontextmanager
from typing import AsyncIterator, cast, Any, LiteralString, override, TYPE_CHECKING
import asyncio
import logging
from abc import ABC, abstractmethod
import os

from psycopg_pool.pool_async import AsyncConnectionPool
from psycopg.cursor_async import AsyncCursor
from psycopg.connection_async import AsyncConnection
from psycopg.rows import TupleRow
from sentence_transformers import SentenceTransformer
from numpy import ndarray

from composer.rag.types import ManualRef, BlockChunk, ManualSectionHit
from composer.rag.text import code_ref_tag

import aiosqlite

_NO_CHROMA = True

if TYPE_CHECKING:
    import chromadb

try:
    import chromadb
except ImportError:
    _NO_CHROMA = True


logger = logging.getLogger(__name__)

# tqdm tries to create a multiprocessing.RLock on first use, which calls
# fork_exec to start a resource tracker process.  In an async event loop
# with open DB connections this fails with "bad value(s) in fds_to_keep"
# and eventually hangs.  Pre-set a threading lock so tqdm never attempts
# the fork.  This is the narrowest fix: no env-var side effects, and
# sentence_transformers (which uses tqdm internally) just works.
import threading
from tqdm import tqdm as _tqdm_cls
_tqdm_cls.set_lock(threading.RLock())

DEFAULT_CONNECTION: str = "postgresql://rag_user:rag_password@localhost:5432/rag_db"
SANITY_DEFAULT_CONNECTION: str = "postgresql://extended_rag_user:rag_password@localhost:5432/extended_rag_db"


type _RagHeader = str | None
type _ContentHeaders = tuple[_RagHeader, _RagHeader, _RagHeader, _RagHeader, _RagHeader, _RagHeader]

class ComposerRAGDB(ABC):
    """Abstract base class for RAG database implementations."""

    def __init__(self, model: SentenceTransformer):
        self.tr = model

    @abstractmethod
    async def add_chunks_batch(self, chunks: list[BlockChunk]) -> None:
        """Add chunks to the database in batches."""
        ...

    @abstractmethod
    async def add_manual_section(self, ch: BlockChunk) -> None:
        """Add a manual section for keyword search and exact retrieval."""
        ...

    @abstractmethod
    async def find_refs(self, query: str, similarity_cutoff: float = 0.5,
                  top_k: int = 10, manual_section: list[str] = []) -> list[ManualRef]:
        """Search for similar documents using semantic similarity."""
        ...

    @abstractmethod
    async def search_manual_keywords(self, query: str, *, min_depth: int = 0,
                               limit: int = 10) -> list[ManualSectionHit]:
        """Search sections by keywords using full-text search."""
        ...

    @abstractmethod
    async def get_manual_section(self, headers: list[str]) -> str | None:
        """Retrieve full section content by exact headers."""
        ...

    
    async def embed_query(
        self, query: str
    ) -> ndarray:
        return cast(ndarray, await asyncio.to_thread(
            self.tr.encode_query, f"search_query: {query}", show_progress_bar=False
        ))

    async def embed_docs(
        self, doc: list[BlockChunk]
    ) -> list[ndarray]:
        return cast(list[ndarray], await asyncio.to_thread(
                self.tr.encode_document, [f"search_document: {d.chunk}" for d in doc], show_progress_bar=False
            ))



type RagConnection = str | AsyncConnectionPool[AsyncConnection[TupleRow]] | AsyncConnection[TupleRow]

class PostgreSQLRAGDatabase(ComposerRAGDB):
    """Handle PostgreSQL database operations for RAG"""

    def __init__(self, conn_string: RagConnection, model: SentenceTransformer):
        super().__init__(model)
        self.conn_string = conn_string
        self._lock = asyncio.Lock() if isinstance(conn_string, AsyncConnection) else None

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
    @staticmethod
    async def rag_context(model: SentenceTransformer, conn_str : str = DEFAULT_CONNECTION) -> AsyncIterator["PostgreSQLRAGDatabase"]:
        async with AsyncConnectionPool(
            conninfo=conn_str,
            kwargs={"autocommit": True},
            connection_class=AsyncConnection[TupleRow]
        ) as pool:
            db = PostgreSQLRAGDatabase(conn_string=pool, model=model)
            await db.test_connection()
            yield db


    @asynccontextmanager
    async def _get_connection(self) -> AsyncIterator[AsyncConnection[TupleRow]]:
        """Get database connection with context manager"""
        if isinstance(self.conn_string, str):
            conn = await AsyncConnection.connect(self.conn_string, autocommit=True)
            async with conn:
                yield conn
        elif isinstance(self.conn_string, AsyncConnection):
            assert self._lock is not None
            async with self._lock:
                try:
                    yield self.conn_string
                    await self.conn_string.commit()
                except:
                    await self.conn_string.rollback()
                    raise
        else:
            async with self.conn_string.connection() as conn:
                yield conn

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

    def _normalize_head(self, l: list[str]) -> _ContentHeaders:
        headers : list[str | None] = [None] * 6
        for (ind, h) in enumerate(l):
            if h:
                headers[ind] = h
        return tuple(headers) #type: ignore[trust me bro]

    @override
    async def add_manual_section(self, ch: BlockChunk):
        async with self._get_connection() as conn:
            async with conn.transaction():
                headers = self._normalize_head(ch.headers)
                data = (ch.chunk,) + headers + (ch.part,)
                cur = await conn.execute("""
                    INSERT INTO manual_sections(
                        content, h1, h2, h3, h4, h5, h6, part
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, data)
                insert_res = await cur.fetchone()
                if insert_res is None:
                    raise Exception("Insertion didn't return ID")
                print(insert_res)
                payloads = []
                for (i, code) in enumerate(ch.code_refs):
                    payloads.append((i, code, insert_res[0]))
                async with conn.cursor() as cur:
                    await cur.executemany("""
                        INSERT INTO manual_section_code_refs(
                            id, code_body, section_id
                        ) VALUES (%s, %s, %s)
                    """, payloads)

    @override
    async def add_chunks_batch(self, chunks: list[BlockChunk]) -> None:
        """Add chunks to database in batches"""
        if not chunks:
            return
        # SentenceTransformer.encode_document is CPU-bound (runs the embedding model);
        # offload to a thread to avoid blocking the event loop.
        embeddings = await self.embed_docs(chunks)

        logger.info(f"Adding {len(chunks)} chunks to database...")
        # Insert batch
        async with self._get_connection() as conn, conn.transaction(), conn.cursor() as cur:
            for chunk, embedding in zip(chunks, embeddings):
                try:
                    headers = self._normalize_head(chunk.headers)
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

    @override
    async def find_refs(self, query: str, similarity_cutoff: float = 0.5, top_k: int = 10, manual_section : list[str] = []) -> list[ManualRef]:
        # SentenceTransformer.encode_query is CPU-bound (runs the embedding model);
        # offload to a thread to avoid blocking the event loop.
        question_embedding = await self.embed_query(query)

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

    async def _replace_manual_code_refs(self, cur: AsyncCursor[TupleRow], content: str, section_id: int) -> str:
        await cur.execute(
            "SELECT id, code_body FROM manual_section_code_refs WHERE section_id = %s",
            (section_id,)
        )
        async for row in cur:
            content = content.replace(code_ref_tag(row[0]), row[1])
        return content

    @override
    async def search_manual_keywords(self, query: str, *, min_depth: int = 0, limit: int = 10) -> list[ManualSectionHit]:
        if min_depth < 0 or min_depth > 6:
            raise ValueError("min_depth must be between 0 and 6")
        depth_clause = f"AND h{min_depth} IS NOT NULL" if min_depth > 0 else ""
        depth_clause = cast(LiteralString, depth_clause)
        async with self._get_connection() as conn:
            cur = await conn.execute(f"""
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

    @override
    async def get_manual_section(self, headers: list[str]) -> str | None:
        padded: list[str | None] = list(headers) + [None] * (6 - len(headers))
        padded = padded[:6]
        async with self._get_connection() as conn:
            cur = await conn.execute("""
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



class ChromaRAGDatabase(ComposerRAGDB):
    """ChromaDB-based RAG database (file-based, no PostgreSQL required)"""

    def __init__(self, persist_dir: str, model: SentenceTransformer):
        super().__init__(model)
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        # Collection for semantic search (embedded chunks)
        self.collection = self.client.get_or_create_collection(
            name="cvl_manual",
            metadata={"hnsw:space": "cosine"}
        )
        # Collection for section storage (exact retrieval by headers)
        self.sections = self.client.get_or_create_collection(
            name="cvl_manual_sections"
        )
        self._next_id = self.collection.count()
        self._next_section_id = self.sections.count()

        # Create FTS5 index for keyword search
        self.fts_conn = aiosqlite.connect(
            os.path.join(persist_dir, "fts_index.db"),
            check_same_thread=False
        )

    @asynccontextmanager
    async def _conn(self) -> AsyncIterator[aiosqlite.Connection]:
        try:
            yield self.fts_conn
            await self.fts_conn.commit()
        except:
            await self.fts_conn.rollback()
            raise

    async def _setup(self):
        self.fts_conn = await self.fts_conn
        async with self._conn() as conn:
            await conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
                    content,
                    h1, h2, h3, h4, h5, h6,
                    section_id UNINDEXED
                )
            """)

    @asynccontextmanager
    @staticmethod
    async def rag_context(persist_dir: str, model: SentenceTransformer):
        to_ret = ChromaRAGDatabase(persist_dir, model)
        await to_ret._setup()
        yield to_ret

    @override
    async def add_chunks_batch(self, chunks: list[BlockChunk]) -> None:
        """Add chunks to ChromaDB"""
        if not chunks:
            return

        embeddings = await self.embed_docs(chunks)

        logger.info(f"Adding {len(chunks)} chunks to ChromaDB...")

        ids = []
        documents = []
        embedding_list = []
        metadatas = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_id = str(self._next_id + i)
            ids.append(doc_id)

            # Expand code refs inline
            body = chunk.chunk
            for j, code in enumerate(chunk.code_refs):
                to_replace = code_ref_tag(j)
                body = body.replace(to_replace, code)
            documents.append(body)

            embedding_list.append(embedding.tolist())

            # Store headers in metadata
            metadata = {}
            for j, header in enumerate(chunk.headers, start=1):
                if header:
                    metadata[f"h{j}"] = header
            metadatas.append(metadata)

        self.collection.add(
            ids=ids,
            embeddings=embedding_list,
            documents=documents,
            metadatas=metadatas
        )
        self._next_id += len(chunks)

    @override
    async def find_refs(self, query: str, similarity_cutoff: float = 0.5,
                  top_k: int = 10, manual_section: list[str] = []) -> list[ManualRef]:
        """Search for similar documents"""
        query_embedding = await self.embed_query(query)

        # Build where filter for manual_section if provided
        where_filter : chromadb.Where | None = None
        if manual_section:
            # ChromaDB uses $or for multiple conditions
            or_conditions : list[chromadb.Where] = []
            for section in manual_section:
                for i in range(1, 7):
                    or_conditions.append({f"h{i}": section})
            if or_conditions:
                where_filter = {"$or": or_conditions}

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        to_ret: list[ManualRef] = []
        if not results['documents'] or not results['documents'][0]:
            return to_ret

        for i, doc in enumerate(results['documents'][0]):
            # ChromaDB returns L2 distance by default, but we configured cosine
            # For cosine distance: similarity = 1 - distance
            distance = results['distances'][0][i] if results['distances'] else 0
            similarity = 1 - distance

            if similarity < similarity_cutoff:
                continue

            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            headers = []
            for j in range(1, 7):
                h = metadata.get(f'h{j}')
                if h:
                    headers.append(h)
                else:
                    break

            to_ret.append(ManualRef(headers=headers, content=doc, similarity=similarity))

        return to_ret

    @override
    async def add_manual_section(self, ch: BlockChunk) -> None:
        """Add a manual section for keyword search and exact retrieval."""
        # Expand code refs inline
        body = ch.chunk
        for j, code in enumerate(ch.code_refs):
            body = body.replace(code_ref_tag(j), code)

        metadata: dict[str, str | int] = {"part": ch.part}
        for i, h in enumerate(ch.headers, start=1):
            if h:
                metadata[f"h{i}"] = h

        self.sections.add(
            ids=[str(self._next_section_id)],
            documents=[body],
            metadatas=[metadata]
        )

        # Also insert into FTS index for keyword search
        headers: list[str | None] = list(ch.headers) + [None] * (6 - len(ch.headers))
        async with self._conn() as conn:
            await conn.execute(
                """INSERT INTO sections_fts(content, h1, h2, h3, h4, h5, h6, section_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (body, *headers[:6], str(self._next_section_id))
            )

        self._next_section_id += 1

    @override
    async def search_manual_keywords(self, query: str, *, min_depth: int = 0, limit: int = 10) -> list[ManualSectionHit]:
        """Search sections by keywords using FTS5 full-text search."""
        if min_depth < 0 or min_depth > 6:
            raise ValueError("min_depth must be between 0 and 6")

        depth_filter = f"AND h{min_depth} IS NOT NULL" if min_depth > 0 else ""

        try:
            async with self._conn() as conn:
                cursor = await conn.execute(f"""
                    SELECT bm25(sections_fts) as rank, h1, h2, h3, h4, h5, h6
                    FROM sections_fts
                    WHERE sections_fts MATCH ?
                    {depth_filter}
                    ORDER BY rank
                    LIMIT ?
                """, (query, limit))


                return [
                    ManualSectionHit(
                        headers=[h for h in row[1:7] if h],
                        relevance=-row[0]  # bm25 returns negative scores, lower = better
                    )
                    for row in await cursor.fetchall()
                ]
        except aiosqlite.OperationalError:
            # Handle invalid FTS query syntax gracefully
            return []

    async def get_manual_section(self, headers: list[str]) -> str | None:
        """Retrieve full section content by exact headers."""
        if not headers:
            return None

        # Build filter for exact header match on provided levels
        conditions : list[chromadb.Where] = []
        for i, h in enumerate(headers, start=1):
            conditions.append({f"h{i}": {"$eq": h}})

        where_filter : chromadb.Where | None = {"$and": conditions} if len(conditions) > 1 else conditions[0] if conditions else None

        results = self.sections.get(
            where=where_filter,
            include=["documents", "metadatas"]
        )

        if not results['documents'] or not results['metadatas']:
            return None

        # Post-filter: exclude subsections where deeper headers exist.
        # ChromaDB can't express "field IS NULL", so we filter in Python.
        # This mirrors Postgres's "hN IS NOT DISTINCT FROM NULL" for N > len(headers).
        depth = len(headers)
        filtered = [
            (doc, meta)
            for doc, meta in zip(results['documents'], results['metadatas'])
            if not any(meta.get(f"h{j}") for j in range(depth + 1, 7))
        ]

        if not filtered:
            return None

        parts = sorted(filtered, key=lambda x: cast(int, x[1].get('part', 0)))
        return "\n".join(doc for doc, _ in parts)

@asynccontextmanager
async def rag_context(
    rag_connection_str: str,
    model: SentenceTransformer
) -> AsyncIterator[ComposerRAGDB]:
    if rag_connection_str.startswith("postgresql://"):
        async with PostgreSQLRAGDatabase.rag_context(model, rag_connection_str) as conn:
            yield conn
    else:
        async with ChromaRAGDatabase.rag_context(persist_dir=rag_connection_str, model=model) as conn:
            yield conn

async def get_rag_db(
    rag_connection_str: str,
    model: SentenceTransformer
) -> ComposerRAGDB:
    if rag_connection_str.startswith("postgresql://"):
        conn = await AsyncConnection.connect(rag_connection_str)
        db = PostgreSQLRAGDatabase(conn, model=model)
        await db.test_connection()
        return db
    else:
        to_ret = ChromaRAGDatabase(model=model, persist_dir=rag_connection_str)
        await to_ret._setup()
        return to_ret
