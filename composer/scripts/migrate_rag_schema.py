#      The Certora Prover
#      Copyright (C) 2025  Certora Ltd.
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, version 3 of the License.
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Migrate the RAG databases to the search_path-based multi-tenancy layout.

Legacy layout: ``rag_user`` lived in ``public`` of ``rag_db``, while
``extended_rag_user`` had its own ``extended_rag_db``; the ``vector`` /
``pg_trgm`` extensions were created in ``public``.

Current layout (``composer/scripts/init-db.sql``): a single ``rag_db`` holding
one schema per tenant (``rag`` / ``extended_rag`` / ``foundry_rag``), each
selected by a per-role ``search_path``, with the extensions hoisted into a
dedicated ``extensions`` schema.

This script brings the target ``rag_db`` up to that layout and rebuilds the RAG
content from the documentation (the RAG is derived entirely from docs, so a
drop + repopulate loses nothing):

  1. ensure the three tenant roles exist;
  2. create the ``extensions`` schema and relocate ``vector`` / ``pg_trgm`` into it;
  3. drop the legacy RAG tables from ``public``;
  4. lock down ``public`` and give each tenant a dedicated schema + search_path;
  5. re-run the cvl, extended, and foundry population scripts.

Every step is existence-checked, so the script is safe to re-run.

Run with::

    python -m composer.scripts.migrate_rag_schema [--admin-dsn ...] [--skip-populate]

The DDL needs a superuser/owner connection (``ALTER EXTENSION``, ``CREATE
SCHEMA ... AUTHORIZATION``, ``ALTER ROLE``); ``--admin-dsn`` defaults to the
docker-compose superuser and should be overridden when running against master.
"""

import argparse
import os
import pathlib
import subprocess
import sys

import psycopg
import psycopg.sql as sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo


# Superuser default mirrors scripts/docker-compose.yml (POSTGRES_USER /
# POSTGRES_PASSWORD); host/port honor the same env vars as composer.rag.db.
# Override --admin-dsn entirely when pointing at a non-local "master" cluster.
_PGHOST = os.environ.get("CERTORA_AI_COMPOSER_PGHOST", "localhost")
_PGPORT = os.environ.get("CERTORA_AI_COMPOSER_PGPORT", "5432")
DEFAULT_ADMIN_DSN = f"postgresql://postgres:postgres_admin_password@{_PGHOST}:{_PGPORT}/rag_db"

# (role, schema) for each tenant; mirrors init-db.sql. Each role's search_path
# becomes "<schema>, extensions" so the lazily-created tables land in its own
# schema and the relocated extensions stay resolvable.
_TENANTS: tuple[tuple[str, str], ...] = (
    ("rag_user", "rag"),
    ("extended_rag_user", "extended_rag"),
    ("foundry_rag_user", "foundry_rag"),
)

# Legacy tables to remove from public; repopulation recreates them per-schema.
_LEGACY_PUBLIC_TABLES = ("manual_section_code_refs", "code_refs", "manual_sections", "documents")

# Populate scripts to re-run, in order. Each connects as its own tenant user, so
# the post-migration search_path routes its tables into the right schema.
_POPULATE_SCRIPTS: tuple[tuple[str, str], ...] = (
    ("CVL manual (rag)", "populate_rag.sh"),
    ("extended RAG (CVL + prover + user-guide)", "populate_extended_rag.sh"),
    ("foundry cheatcodes (foundry_rag)", "populate_foundry_rag.sh"),
)

_CONFIRMATION = "Yes, migrate the rag schema"


def _ensure_role(cur: psycopg.Cursor, role: str) -> None:
    """CREATE the tenant role if it's missing (password matches init-db.sql)."""
    cur.execute("SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = %s", (role,))
    if cur.fetchone() is None:
        cur.execute(
            sql.SQL("CREATE USER {} WITH PASSWORD {}").format(
                sql.Identifier(role), sql.Literal("rag_password")
            )
        )


def _relocate_extension(cur: psycopg.Cursor, ext: str) -> None:
    """Ensure ``ext`` lives in the extensions schema: create it there, or move it."""
    cur.execute(
        """
        SELECT n.nspname FROM pg_extension e
        JOIN pg_namespace n ON n.oid = e.extnamespace
        WHERE e.extname = %s
        """,
        (ext,),
    )
    row = cur.fetchone()
    if row is None:
        cur.execute(sql.SQL("CREATE EXTENSION {} SCHEMA extensions").format(sql.Identifier(ext)))
    elif row[0] != "extensions":
        cur.execute(sql.SQL("ALTER EXTENSION {} SET SCHEMA extensions").format(sql.Identifier(ext)))


def _setup_tenant(cur: psycopg.Cursor, db_name: str, role: str, schema: str) -> None:
    """Create the tenant's schema, grant extension access, and pin its search_path."""
    role_id, schema_id, db_id = sql.Identifier(role), sql.Identifier(schema), sql.Identifier(db_name)
    cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {} AUTHORIZATION {}").format(schema_id, role_id))
    cur.execute(sql.SQL("GRANT USAGE ON SCHEMA extensions TO {}").format(role_id))
    cur.execute(
        sql.SQL("ALTER ROLE {} IN DATABASE {} SET search_path = {}, extensions").format(
            role_id, db_id, schema_id
        )
    )


def run_ddl(admin_dsn: str) -> None:
    """Apply the migration DDL in a single transaction (all-or-nothing)."""
    with psycopg.connect(admin_dsn, autocommit=False) as conn:
        db_name = conn.info.dbname
        with conn.cursor() as cur:
            print("  • ensure tenant roles exist")
            for role, _ in _TENANTS:
                _ensure_role(cur, role)

            print("  • create extensions schema")
            cur.execute("CREATE SCHEMA IF NOT EXISTS extensions")

            for table in _LEGACY_PUBLIC_TABLES:
                print(f"  • drop legacy public.{table}")
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS public.{} CASCADE").format(sql.Identifier(table))
                )

            for ext in ("vector", "pg_trgm"):
                print(f"  • relocate {ext} extension into extensions schema")
                _relocate_extension(cur, ext)

            print("  • lock down public schema")
            cur.execute("REVOKE ALL ON SCHEMA public FROM PUBLIC")

            for role, schema in _TENANTS:
                print(f"  • set up schema '{schema}' for {role}")
                _setup_tenant(cur, db_name, role, schema)
        conn.commit()
    print(f"DDL applied to database '{db_name}'.")


def run_population(skip: bool) -> None:
    if skip:
        print("\nSkipping population (--skip-populate). Run the populate_*.sh scripts manually when ready.")
        return
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "scripts"
    for label, script in _POPULATE_SCRIPTS:
        path = scripts_dir / script
        print(f"\n==> Populating {label}: bash {path}")
        result = subprocess.run(["bash", str(path)], cwd=repo_root)
        if result.returncode != 0:
            raise SystemExit(f"Population step '{script}' failed (exit {result.returncode}).")


def _redact(dsn: str) -> str:
    parts = conninfo_to_dict(dsn)
    if parts.get("password"):
        parts["password"] = "***"
    return make_conninfo("", **parts)


def _confirm(admin_dsn: str) -> bool:
    print()
    print("=" * 72)
    print("  ABOUT TO MIGRATE THE RAG DATABASE SCHEMA")
    print("=" * 72)
    print(f"  Admin connection: {_redact(admin_dsn)}")
    print("  This will DROP the legacy RAG tables in public:")
    for t in _LEGACY_PUBLIC_TABLES:
        print(f"    - public.{t}")
    print("  relocate the vector/pg_trgm extensions, repartition the tenant")
    print("  roles into dedicated schemas, and rebuild the RAG content.")
    print()
    print("  Type EXACTLY the following confirmation to proceed:")
    print()
    print(f"      {_CONFIRMATION}")
    print()
    return input("> ").strip() == _CONFIRMATION


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="migrate_rag_schema",
        description="Migrate rag_db to the search_path-based multi-tenancy layout, then repopulate.",
    )
    parser.add_argument(
        "--admin-dsn",
        default=DEFAULT_ADMIN_DSN,
        help=f"Superuser/owner DSN for the DDL (default: {_redact(DEFAULT_ADMIN_DSN)}). "
             "Override when running against master.",
    )
    parser.add_argument(
        "--skip-populate",
        action="store_true",
        help="Stop after the DDL; do not re-run the populate_*.sh scripts.",
    )
    parser.add_argument(
        "--skip-confirmation",
        action="store_true",
        help="Bypass the interactive typed-confirmation guard.",
    )
    args = parser.parse_args()

    if not args.skip_confirmation and not _confirm(args.admin_dsn):
        print("\nConfirmation did not match. Aborting; no changes made.")
        return 1

    run_ddl(args.admin_dsn)
    run_population(args.skip_populate)
    print("\n✅ RAG schema migration complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
