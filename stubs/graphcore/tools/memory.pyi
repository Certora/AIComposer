import pathlib
import sqlite3

from psycopg import Connection

from langchain_core.tools import BaseTool

class MemoryBackend:
    ...

class PostgresMemoryBackend(MemoryBackend):
    def __init__(self, ns: str, conn: Connection, init_from: str | None = ...):
        ...

class SqliteMemoryBackend(MemoryBackend):
    def __init__(self, ns: str, conn: sqlite3.Connection, init_from: str | None = ...):
        ...

class FileSystemMemoryBackend(MemoryBackend):
    def __init__(self, storage_folder: pathlib.Path, init_from: pathlib.Path | None = ...):
        ...

def memory_tool(backend: MemoryBackend) -> BaseTool:
    ...
