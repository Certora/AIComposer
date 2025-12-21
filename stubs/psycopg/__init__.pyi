from typing import TypeVar, overload
import psycopg.connection as conn
from psycopg.rows import RowFactory, TupleRow
from psycopg.cursor import Cursor as C

T = TypeVar('T', covariant=True)

Connection = conn.Connection

Cursor = C

@overload
def connect(conn: str, autocommit: bool = ...) -> Connection[TupleRow]:
    ...

@overload
def connect(conn: str, autocommit: bool, row_factory: RowFactory[T]) -> Connection[T]:
    ...
