import psycopg
from psycopg import Connection
from typing import Optional, Iterable, Tuple, cast, Iterator, Literal, Callable, LiteralString
from typing_extensions import Iterable
import hashlib
import pathlib
import gzip
from dataclasses import dataclass
from functools import cached_property

from verisafe.rag.types import ManualRef
from verisafe.audit.types import ManualResult, RuleResult, RunInput, InputFileLike

DEFAULT_CONNECTION = "postgresql://audit_db_user:audit_db_password@localhost:5432/audit_db"

_resume_q : LiteralString = """
SELECT 
 r.commentary,
 r.interface_path,
 ri.system_id as system_id,
 ri.system_name as system_name,
 final_spec_id.file_id as spec_id,
 intf_id.file_id as interface_id
FROM resume_artifact r
INNER JOIN vfs_result intf_id ON intf_id.path = r.interface_path AND intf_id.thread_id = r.thread_id
INNER JOIN run_info ri ON ri.thread_id = r.thread_id
INNER JOIN vfs_result final_spec_id ON r.thread_id = final_spec_id.thread_id AND final_spec_id.path = 'rules.spec'
WHERE r.thread_id = %s
"""

_VFSTable = Literal["vfs_result", "vfs_initial"]

class VFSFile:
    def __init__(self, path: str, file_id: str, conn: Connection):
        self.conn = conn
        self.path = path
        self.file_id = file_id

    @property
    def basename(self) -> str:
        return pathlib.Path(self.path).name

    @property
    def bytes_contents(self) -> bytes:
        with self.conn.transaction():
            with self.conn.cursor() as cur:
                cur.execute("""
SELECT file_blob from file_blobs WHERE file_id = %s
""", (self.file_id,))
                result = cur.fetchone()
                if result is None:
                    raise RuntimeError(f"File {self.file_id} not found")
                return gzip.decompress(bytes(result[0]))
        
    @property
    def string_contents(self) -> str:
        return self.bytes_contents.decode("utf-8")


@dataclass
class VFSRetriever:
    _table: _VFSTable
    thread_id: str
    conn: Connection

    def to_dict(self) -> dict[str, bytes]:
        to_ret = {}
        for (p, cont) in self:
            to_ret[p] = cont
        return to_ret

    def __iter__(self) -> Iterator[tuple[str, bytes]]:
        with self.conn.transaction():
            with self.conn.cursor() as cur:
                cur.execute(f"""
SELECT
   path,
   f.file_blob
FROM {self._table} t
INNER JOIN file_blobs f ON f.file_id = t.file_id
WHERE thread_id = %s
""", (self.thread_id,))
                for r in cur:
                    p = cast(str, r[0])
                    compressed_blob = bytes(r[1])
                    yield (p, gzip.decompress(compressed_blob))

    def get_file(self, p: str) -> VFSFile | None:
        with self.conn.transaction():
            with self.conn.cursor() as cur:
                cur.execute(f"""
SELECT file_id FROM {self._table} WHERE path = %s AND thread_id = %s
""", (p, self.thread_id))
                r = cur.fetchone()
                if r is None:
                    return None
                return VFSFile(p, r[0], self.conn)

    def __getitem__(self, p: str) -> VFSFile | None:
        return self.get_file(p)


class ResumeArtifact:
    def __init__(self,
                 final_intf: VFSFile,
                 final_spec: VFSFile,
                 system_doc: VFSFile,
                 commentary: str,
                 intf_path: str,
                 vfs_cur: VFSRetriever):
        self.intf_vfs_handle = final_intf
        self.spec_vfs_handle = final_spec
        self.system_vfs_handle = system_doc
        self.vfs = vfs_cur
        self.commentary = commentary
        self.interface_path = intf_path

    @cached_property
    def interface_file(self) -> str:
        return self.intf_vfs_handle.bytes_contents.decode("utf-8")
    
    @cached_property
    def spec_file(self) -> str:
        return self.spec_vfs_handle.bytes_contents.decode("utf-8")
    
    @cached_property
    def system_doc(self) -> str:
        return self.system_vfs_handle.bytes_contents.decode("utf-8")
        

class AuditDB():
    class _StringFile:
        def __init__(self, path: str, contents: str):
            self.path = path
            self.contents = contents

        @property
        def bytes_contents(self) -> bytes:
            return self.contents.encode("utf-8")
        
        @property
        def basename(self) -> str:
            return pathlib.Path(self.path).name

    def __init__(self, conn: Connection):
        self.conn = conn
    
    def add_rule_result(self, thread_id: str, tool_id: str, rule_name: str, result: str, analysis: Optional[str]):
        with self.conn.transaction():
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO prover_results(tool_id, rule_name, thread_id, result, analysis) VALUES
                    (%s, %s, %s, %s, %s)
                    ON CONFLICT (tool_id, rule_name, thread_id) DO UPDATE
                    SET result = EXCLUDED.result, analysis = EXCLUDED.analysis
                """, (tool_id, rule_name, thread_id, result, analysis))
    
    def add_manual_result(self, thread_id: str, tool_id: str, ref: ManualRef):
        with self.conn.transaction():
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO manual_results(tool_id, thread_id, similarity, text_body, header_string)
                    VALUES (%s, %s, %s, %s, %s)
                """, (tool_id, thread_id, ref.similarity, ref.content, " / ".join(ref.headers)))

    def get_rule_results(self, thread_id: str, tool_id: str) -> Iterable[RuleResult]:
        with self.conn.transaction():
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT rule_name, result, analysis FROM prover_results WHERE tool_id = %s AND thread_id = %s
                """, (tool_id, thread_id))
                for row in cur:
                    yield RuleResult(analysis=row[2], rule=row[0], status=row[1])

    def get_manual_results(self, thread_id: str, tool_id: str) -> Iterable[ManualResult]:
        with self.conn.transaction():
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT header_string, text_body, similarity FROM manual_results
                    WHERE thread_id = %s AND tool_id = %s
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
    
    def _prepare_blobs(self, thread_id: str, table: _VFSTable, vfs: Iterable[tuple[str, bytes]]) -> Callable[[psycopg.Cursor], None]:
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

        def thunk(cur: psycopg.Cursor):
            cur.executemany(
                """
INSERT INTO file_blobs(file_id, file_blob) VALUES (%s, %s)
ON CONFLICT (file_id) DO NOTHING
""",
                blob_updates
            )
            cur.executemany(f"""
INSERT INTO {table}(
    thread_id, path, file_id
) VALUES (%s, %s, %s)
ON CONFLICT (thread_id, path) DO UPDATE
SET file_id = EXCLUDED.file_id
""",
                file_updates
            )
        return thunk


    def register_run(self,
                     thread_id: str,
                     spec_file: InputFileLike,
                     interface_file: InputFileLike,
                     system_doc: InputFileLike,
                     vfs_init: Iterable[tuple[str, bytes]],
                     reqs: list[str] | None
                     ) -> None:
        (spec_hash, spec_compress) = self._hash_and_compress(spec_file)
        (interface_hash, interface_compress) = self._hash_and_compress(interface_file)
        (system_hash, system_compress) = self._hash_and_compress(system_doc)
        vfs_thunk = self._prepare_blobs(
            thread_id=thread_id,
            table="vfs_initial",
            vfs=vfs_init
        )
        r_list = reqs if reqs is not None else []
        with self.conn.transaction():
            with self.conn.cursor() as cur:
                cur.executemany("""
                    INSERT INTO file_blobs(file_id, file_blob) VALUES (%s, %s)
                    ON CONFLICT (file_id) DO NOTHING
                """, [
                    (spec_hash, spec_compress),
                    (interface_hash, interface_compress),
                    (system_hash, system_compress)
                ])
                cur.execute("""
                    INSERT INTO run_info(thread_id, spec_id, spec_name, interface_id, interface_name, system_id, system_name, num_reqs)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (thread_id) DO UPDATE
                    SET spec_id = EXCLUDED.spec_id, spec_name = EXCLUDED.spec_name,
                        interface_id = EXCLUDED.interface_id, interface_name = EXCLUDED.interface_name,
                        system_id = EXCLUDED.system_id, system_name = EXCLUDED.system_name,
                        num_reqs = EXCLUDED.num_reqs
                """, (thread_id, 
                      spec_hash,
                      spec_file.basename,
                      interface_hash,
                      interface_file.basename,
                      system_hash,
                      system_doc.basename,
                      len(reqs) if reqs is not None else None))
                vfs_thunk(cur)
                cur.executemany("""
                    INSERT INTO run_requirements(thread_id, req_num, req_text) VALUES (%s, %s, %s)
                    ON CONFLICT (thread_id,req_num) DO UPDATE
                    SET req_text = EXCLUDED.req_text
                """, [
                    (thread_id, i + 1, req) for (i, req) in enumerate(r_list)
                ])

    def register_complete(self, thread_id: str, vfs: Iterable[tuple[str, bytes]], intf: str, commentary: str) -> None:
        vfs_update_thunk = self._prepare_blobs(thread_id=thread_id, table="vfs_result", vfs=vfs)
        with self.conn.transaction():
            with self.conn.cursor() as cur:
                vfs_update_thunk(cur)
                cur.execute(
                    """
INSERT INTO resume_artifact(thread_id, interface_path, commentary) VALUES (%s, %s, %s)
ON CONFLICT (thread_id) DO UPDATE
SET interface_path = EXCLUDED.interface_path, commentary = EXCLUDED.commentary
""", (thread_id, intf, commentary)
                )

    def get_resume_artifact(self, thread_id: str) -> ResumeArtifact:
        with self.conn.transaction():
            with self.conn.cursor() as cur:
                retriever = VFSRetriever(_table="vfs_result", thread_id=thread_id, conn=self.conn)
                cur.execute(_resume_q, (thread_id,))
                r = cur.fetchone()
                if r is None:
                    raise RuntimeError(f"No resume artifact found for thread {thread_id}")

                system_file = VFSFile(
                    file_id=r[2],
                    path=r[3],
                    conn=self.conn
                )
                interface_file = VFSFile(
                    file_id=r[5],
                    conn=self.conn,
                    path=r[1]
                )

                rule_file = VFSFile(
                    conn=self.conn,
                    path="rules.spec",
                    file_id=r[4]
                )

                return ResumeArtifact(
                    commentary=r[0],
                    intf_path=r[1],
                    final_intf=interface_file,
                    final_spec=rule_file,
                    system_doc=system_file,
                    vfs_cur=retriever
                )
        
    def register_summary(self, thread_id: str, checkpoint_id: str, summary: str):
        with self.conn.transaction():
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO summarization(thread_id, checkpoint_id, summary) VALUES (%s, %s, %s)
                    ON CONFLICT (thread_id, checkpoint_id) DO UPDATE
                    SET summary = EXCLUDED.summary
                """, (thread_id, checkpoint_id, summary))
    
    def get_summary_after_checkpoint(self, thread_id: str, checkpoint_id: str) -> str | None:
        with self.conn.transaction():
            with self.conn.cursor() as cur:
                cur.execute("SELECT summary FROM summarization WHERE thread_id = %s AND checkpoint_id = %s", (thread_id, checkpoint_id))
                r = cur.fetchone()
                if r is not None:
                    return r[0]
                return None

    def get_run_info(self, thread_id: str) -> tuple[RunInput, VFSRetriever]:
        vfs_accessor = VFSRetriever(
            _table="vfs_initial",
            conn=self.conn,
            thread_id=thread_id
        )

        with self.conn.transaction():
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        r.spec_name,
                        r.spec_id,
                        r.interface_name,
                        r.interface_id,
                        r.system_name,
                        r.system_id,
                        r.num_reqs
                    FROM run_info r
                    WHERE r.thread_id = %s
                """, (thread_id,))
                r = cur.fetchone()
                if r is None:
                    raise RuntimeError(f"Didn't find run info for {thread_id}")

                spec_name = r[0]
                spec_id = r[1]

                interface_name = r[2]
                interface_id = r[3]

                sys_name = r[4]
                sys_id = r[5]

                num_reqs = r[6]

                reqs : list[str] | None= None
                if num_reqs is not None:
                    cur.execute("""
SELECT req_text from run_requirements WHERE thread_id = %s ORDER BY req_num ASC LIMIT %s
                                """, (thread_id, num_reqs))
                    reqs = [ r[0] for r in cur ]
                    assert len(reqs) == num_reqs, "Missing requirements"

                return (RunInput(
                    interface=VFSFile(interface_name, interface_id, self.conn),
                    spec=VFSFile(
                        spec_name,
                        spec_id,
                        self.conn
                    ),
                    system=VFSFile(
                        sys_name,
                        sys_id,
                        self.conn
                    ),
                    reqs=reqs
                ), vfs_accessor)
