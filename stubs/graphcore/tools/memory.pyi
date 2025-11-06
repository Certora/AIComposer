import pathlib
import sqlite3

from psycopg import Connection

from langchain_core.tools import BaseTool

class MemoryBackend:
    ...

class PostgresMemoryBackend(MemoryBackend):
    def __init__(self, ns: str, conn: Connection):
        ...

class SqliteMemoryBackend(MemoryBackend):
    def __init__(self, ns: str, conn: sqlite3.Connection):
        ...

class FileSystemMemoryBackend(MemoryBackend):
    def __init__(self, storage_folder: pathlib.Path):
        ...

def memory_tool(backend: MemoryBackend) -> BaseTool:
    ...
