import sqlite3
from typing import Optional, Iterable, Tuple, cast
from typing_extensions import Iterable
import hashlib
import gzip
from dataclasses import dataclass

from verisafe.rag.types import ManualRef
from verisafe.audit.types import ManualResult, RuleResult, RunInput, InputFileLike, InputFile

_setup_script ="""
    CREATE TABLE IF NOT EXISTS prover_results(
        tool_id TEXT NOT NULL,
        rule_name TEXT NOT NULL,
        thread_id TEXT NOT NULL,
        result TEXT NOT NULL CHECK (result in ('VIOLATED', 'ERROR', 'TIMEOUT', 'VERIFIED')),
        analysis TEXT,
        CONSTRAINT pk PRIMARY KEY (tool_id, rule_name, thread_id) ON CONFLICT REPLACE
    );

    CREATE TABLE IF NOT EXISTS manual_results(
        tool_id TEXT NOT NULL,
        thread_id TEXT NOT NULL,
        similarity FLOAT NOT NULL,
        text_body TEXT NOT NULL,
        header_string TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS manual_result_idx ON manual_results (tool_id, thread_id);

    CREATE TABLE IF NOT EXISTS file_blobs(
        file_id VARCHAR(64) PRIMARY KEY ON CONFLICT IGNORE,
        file_blob BLOB NOT NULL
    );

    CREATE TABLE IF NOT EXISTS run_info(
        thread_id TEXT NOT NULL PRIMARY KEY ON CONFLICT REPLACE,
        spec_id VARCHAR(64) NOT NULL REFERENCES file_blobs(file_id),
        spec_name TEXT NOT NULL,
        interface_id VARCHAR(64) NOT NULL REFERENCES file_blobs(file_id),
        interface_name TEXT NOT NULL,
        system_id VARCHAR(64) NOT NULL REFERENCES file_blobs(file_id),
        system_name TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS vfs_result(
        thread_id TEXT NOT NULL REFERENCES run_info(thread_id),
        path TEXT NOT NULL,
        file_id VARCHAR(64) REFERENCES file_blobs(file_id),
        CONSTRAINT thread_path_pk PRIMARY KEY(thread_id, path) ON CONFLICT REPLACE
    );

    CREATE INDEX IF NOT EXISTS vfs_thread_idx on vfs_result(thread_id);
"""

class AuditDB():
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._setup()

    def _setup(self) -> None:
        with self.conn:
            cur = self.conn.cursor()
            cur.executescript(_setup_script)
    
    def add_rule_result(self, thread_id: str, tool_id: str, rule_name: str, result: str, analysis: Optional[str]):
        with self.conn:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO prover_results(tool_id, rule_name, thread_id, result, analysis) VALUES
                (?, ?, ?, ?, ?)
            """, (tool_id, rule_name, thread_id, result, analysis))
    
    def add_manual_result(self, thread_id: str, tool_id: str, ref: ManualRef):
        with self.conn:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO manual_results(tool_id, thread_id, similarity, text_body, header_string)
                VALUES (?, ?, ?, ?, ?)
            """, (tool_id, thread_id, ref.similarity, ref.content, " / ".join(ref.headers)))

    def get_rule_results(self, thread_id: str, tool_id: str) -> Iterable[RuleResult]:
        cur = self.conn.cursor()
        cur.execute("""
            SELECT rule_name, result, analysis FROM prover_results WHERE tool_id = ? AND thread_id = ?
        """, (tool_id, thread_id))
        for row in cur:
            yield RuleResult(analysis=row[2], rule=row[0], status=row[1])

    def get_manual_results(self, thread_id: str, tool_id: str) -> Iterable[ManualResult]:
        cur = self.conn.cursor()
        cur.execute("""
            SELECT header_string, text_body, similarity FROM manual_results
            WHERE thread_id = ? AND tool_id = ?
        """, (thread_id, tool_id))
        for row in cur:
            yield ManualResult(header=row[0], content=row[1], similarity=row[2])

    def _hash_and_compress_bytes(self, b: bytes) -> Tuple[str, bytes]:
        f_bytes = b
        f_hash = hashlib.sha256(f_bytes).hexdigest()
        f_compress = gzip.compress(f_bytes, mtime=None)
        return (f_hash, f_compress)

    def _hash_and_compress(self, f: InputFileLike) -> Tuple[str, bytes]:
        return self._hash_and_compress_bytes(f.bytes_contents)

    def register_run(self, thread_id: str, spec_file: InputFileLike, interface_file: InputFileLike, system_doc: InputFileLike) -> None:
        (spec_hash, spec_compress) = self._hash_and_compress(spec_file)
        (interface_hash, interface_compress) = self._hash_and_compress(interface_file)
        (system_hash, system_compress) = self._hash_and_compress(system_doc)
        with self.conn:
            cur = self.conn.cursor()
            cur.executemany("""
                INSERT INTO file_blobs(file_id, file_blob) VALUES (?, ?)
            """, [
                (spec_hash, spec_compress),
                (interface_hash, interface_compress),
                (system_hash, system_compress)
            ])
            cur.execute("""
                INSERT INTO run_info(thread_id, spec_id, spec_name, interface_id, interface_name, system_id, system_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (thread_id, spec_hash, spec_file.basename, interface_hash, interface_file.basename,  system_hash, system_doc.basename))

    def register_complete(self, thread_id: str, vfs: Iterable[tuple[str, bytes]]) -> None:
        files = [
            (lambda d: (nm, d[0], d[1]))(self._hash_and_compress_bytes(cont))
            for (nm, cont) in vfs
        ]

        blob_updates = [
            (r[1], r[2]) for r in files
        ]
        
        file_updates = [
            (thread_id, r[0], r[1]) for r in files
        ]
        with self.conn:
            cur = self.conn.cursor()
            cur.executemany(
                """
INSERT INTO file_blobs(file_id, file_blob) VALUES (?, ?)
""",
blob_updates
            )
            cur.executemany(
                """
INSERT INTO vfs_result(
    thread_id, path, file_id
) VALUES (?, ?, ?)
""",
file_updates
            )

    def get_run_info(self, thread_id: str) -> RunInput:
        cur = self.conn.cursor()
        cur.execute("""
            SELECT
                r.spec_name, sp.file_blob as spec_contents,
                r.interface_name, i.file_blob as interface_contents,
                r.system_name, sys.file_blob as interface_contents
            FROM run_info r 
            INNER JOIN file_blobs sp ON sp.file_id = r.spec_id
            INNER JOIN file_blobs i ON i.file_id = r.interface_id
            INNER JOIN file_blobs sys ON sys.file_id = r.system_id
            WHERE r.thread_id = ?
        """, (thread_id,))

        for r in cur:
            spec_name = r[0]
            spec_contents = gzip.decompress(cast(bytes, r[1])).decode("utf-8")

            interface_name = r[2]
            interface_contents = gzip.decompress(r[3]).decode("utf-8")

            sys_name = r[4]
            sys_contents = gzip.decompress(r[5]).decode("utf-8")
            return RunInput(
                interface=InputFile(
                    content=interface_contents,
                    basename=interface_name
                ),
                spec=InputFile(
                    basename=spec_name,
                    content=spec_contents
                ),
                system=InputFile(
                    basename=sys_name,
                    content=sys_contents
                )
            )
        raise RuntimeError(f"Didn't find run info for {thread_id}")