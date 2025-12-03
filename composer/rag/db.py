from contextlib import contextmanager
from typing import Generator, List, cast, Any
import logging

import psycopg
from sentence_transformers import SentenceTransformer
from numpy import ndarray

from composer.rag.types import ManualRef, BlockChunk
from composer.rag.text import code_ref_tag

logger = logging.getLogger(__name__)


DEFAULT_CONNECTION: str = "postgresql://rag_user:rag_password@localhost:5432/rag_db"

class PostgreSQLRAGDatabase:
    """Handle PostgreSQL database operations for RAG"""

    def __init__(self, conn_string: str, model: SentenceTransformer, skip_test : bool = True):
        self.conn_string = conn_string
        self.tr = model
        # Test connection
        if not skip_test:
            self._test_connection()

    def _test_connection(self) -> None:
        """Test database connection and setup"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    logger.info("✅ Database connection successful")

                    # Check if documents table exists
                    cur.execute("""
                        SELECT 1 FROM information_schema.tables
                        WHERE table_name = 'documents'
                    """)
                    if not cur.fetchone():
                        logger.warning("❌ Documents table not found, creating...")
                        self._create_schema()
                    else:
                        logger.info("✅ Documents table found")

        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    @contextmanager
    def _get_connection(self) -> Generator[psycopg.Connection, None, None]:
        """Get database connection with context manager"""
        conn = None
        try:
            conn = psycopg.connect(self.conn_string)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()

    def _create_schema(self) -> None:
        """Create database schema"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # create vector extension
                cur.execute("""
                    CREATE EXTENSION IF NOT EXISTS vector;
                """)

                # Create documents table
                cur.execute("""
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
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS documents_embedding_idx
                    ON documents USING hnsw (embedding vector_cosine_ops);
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS code_refs_lkp ON code_refs(parent_doc);
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS section_h1 ON documents (h1);
                    CREATE INDEX IF NOT EXISTS section_h2 ON documents (h2);
                    CREATE INDEX IF NOT EXISTS section_h3 ON documents (h3);
                    CREATE INDEX IF NOT EXISTS section_h4 ON documents (h4);
                """)

                cur.execute("CREATE INDEX IF NOT EXISTS documents_content_idx ON documents USING gin(to_tsvector('english', content));")

                conn.commit()
                logger.info("✅ Database schema created successfully")

    def add_chunks_batch(self, chunks: List[BlockChunk]) -> None:
        """Add chunks to database in batches"""
        if not chunks:
            return
        embeddings = cast(List[ndarray], self.tr.encode_document([d.chunk for d in chunks], show_progress_bar=False))

        logger.info(f"Adding {len(chunks)} chunks to database...")
        # Insert batch
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for chunk, embedding in zip(chunks, embeddings):
                    try:
                        headers = tuple([f if f else None for f in chunk.headers])
                        payload = (chunk.chunk, embedding.tolist()) + headers
                        cur.execute("""
                            INSERT INTO documents
                            (content, embedding, h1, h2, h3, h4, h5, h6)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            RETURNING id
                        """, payload)
                        insert_res = cur.fetchone()
                        if insert_res is None:
                            raise Exception("Insertion didn't return any data")
                        new_id = insert_res[0]
                        for (i, code) in enumerate(chunk.code_refs):
                            cur.execute("""
                                INSERT INTO code_refs (ref_number, code_body, parent_doc) VALUES (%s, %s, %s)
                            """, (
                                i, code, new_id
                            ))
                    except Exception as e:
                        logger.error(f"Failed to insert chunk {chunk}: {e}")
                        continue

                conn.commit()

    def find_refs(self, query: str, similarity_cutoff: float = 0.5, top_k: int = 10, manual_section : List[str] = []) -> List[ManualRef]:
        question_embedding = cast(ndarray, self.tr.encode_query(query, show_progress_bar=False))

        params: tuple[Any, ...] = (question_embedding.tolist(),)
        where_clause = ""
        if len(manual_section) > 0:
            clauses = []
            for i in range(1, 7):
                params = params + (tuple(manual_section),)
                clauses.append(f"h{i} in %s")
            where_clause = "WHERE (" + " OR ".join(clauses) + ")"
        params += (question_embedding.tolist(), top_k)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT id, content, 1 - (embedding <=> %s::vector) AS cosine_similarity, h1, h2, h3, h4, h5, h6
                    FROM documents
                    {where_clause}
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s
                """, params)

                res = cur.fetchall()
                to_ret = []

                for row in res:
                    body : str = row[1]
                    similarity = row[2]
                    if similarity < similarity_cutoff:
                        break
                    header: List[str] = []
                    for i in row[3:]:
                        if i is None:
                            break
                        assert isinstance(i, str)
                        header.append(i)
                    cur.execute(
                        """
                            SELECT ref_number, code_body FROM
                            code_refs WHERE parent_doc = %s
                        """, (row[0], )
                    )
                    for row in cur:
                        id = row[0]
                        to_replace = code_ref_tag(id)
                        body = body.replace(to_replace, row[1])
                    to_ret.append(ManualRef(headers=header, content=body, similarity=similarity))
                return to_ret
