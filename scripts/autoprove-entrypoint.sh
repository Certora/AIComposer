#!/usr/bin/env bash
# Entrypoint for the autoprove container.
#
# Two responsibilities:
#   1. One-time `setup-db` subcommand — populates rag_db and the LangGraph
#      knowledge base against the compose-managed postgres.
#   2. For console-autoprove / tui-autoprove, transparently inject --rag-db
#      pointing at the in-network postgres service so users don't have to
#      pass the connection string on every run.

set -euo pipefail

: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY must be set in the container env}"
: "${AUTOPROVE_HOME:?AUTOPROVE_HOME not set (image misconfigured)}"

# Compose runs the container as the host's UID/GID via `user:`. That UID is
# almost never in /etc/passwd, and libraries that call pwd.getpwuid() (e.g.
# torch via getpass.getuser()) crash with `KeyError: 'getpwuid(): uid not
# found: <UID>'`. The image's build step makes /etc/passwd and /etc/group
# world-writable so we can append a synthetic entry here.
_uid=$(id -u)
_gid=$(id -g)
if ! getent passwd "$_uid" >/dev/null 2>&1; then
  echo "autoprove:x:${_uid}:${_gid}:autoprove:${HOME}:/bin/bash" >> /etc/passwd
fi
if ! getent group "$_gid" >/dev/null 2>&1; then
  echo "autoprove:x:${_gid}:" >> /etc/group
fi
export USER=autoprove LOGNAME=autoprove

PGHOST="${CERTORA_AI_COMPOSER_PGHOST:-postgres}"
PGPORT="${CERTORA_AI_COMPOSER_PGPORT:-5432}"
RAG_CONN="postgresql://rag_user:rag_password@${PGHOST}:${PGPORT}/rag_db"

if [[ "${1:-}" == "setup-db" ]]; then
  shift
  cd "$AUTOPROVE_HOME/AIComposer"
  echo "[autoprove] populating rag_db at ${RAG_CONN} ..."
  # populate_rag.sh hard-codes localhost via DEFAULT_CONNECTION. Inside the
  # container we need to target the `postgres` service on the docker network,
  # so call ragbuild.py directly with --output instead.
  uv run --isolated --group ragbuild -s scripts/ragbuild.py \
      --output "$RAG_CONN" \
      scripts/prover-docs/cvl.html
  echo "[autoprove] populating LangGraph knowledge base ..."
  uv run --extra ml -s scripts/kb_populate.py
  echo "[autoprove] setup-db done."
  exit 0
fi

# For the prove entry points, inject --rag-db if the user didn't supply one.
case "${1:-}" in
  console-autoprove|tui-autoprove)
    cmd="$1"; shift
    has_rag_db=0
    for arg in "$@"; do
      if [[ "$arg" == "--rag-db" || "$arg" == --rag-db=* ]]; then
        has_rag_db=1
        break
      fi
    done
    if (( has_rag_db == 0 )); then
      set -- "$@" --rag-db "$RAG_CONN"
    fi
    exec "$cmd" "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
