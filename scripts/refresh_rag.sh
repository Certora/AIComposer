#!/bin/bash
# refresh_rag.sh — rebuild the CVL-manual RAG from the published docs, end to end.
#
# The RAG is read-only at runtime and 100% derived from the documentation HTML,
# so a full wipe+rebuild loses nothing. This orchestrates the three steps you'd
# otherwise run by hand:
#
#   1. (re)generate the docs HTML      -> gen_docs.sh
#   2. wipe the target RAG database    -> wipe_rag.py   (auto-confirmed, see below)
#   3. rebuild it from the fresh HTML  -> populate_rag.sh / populate_extended_rag.sh
#
# Run this OFFLINE (when nothing is querying the RAG) — step 2 empties the tables
# and step 3 takes a few minutes to re-embed, during which search would return
# nothing. If you need zero-downtime updates instead, see
# cvl_rag_update_investigation.html for the per-source / staging-swap designs.
#
# It auto-confirms wipe_rag.py's typed-confirmation guard (by piping the sentinel
# read straight out of wipe_rag.py) because here the wipe is immediately followed
# by a rebuild from freshly generated source — it is not a standalone destructive
# op. wipe_rag.py keeps its guard for direct, manual use.
set -euo pipefail

script_dir="$(realpath "$(dirname "$0")")"
parent="$(realpath "$script_dir/..")"

# extended_rag_db connection (mirrors populate_extended_rag.sh / SANITY_DEFAULT_CONNECTION)
ext_conn="postgresql://extended_rag_user:rag_password@localhost:5432/extended_rag_db"

# --- options (defaults: regenerate docs, refresh the CVL-only rag_db) ---
do_gen=1        # 0 with --skip-gen-docs to reuse existing scripts/prover-docs/*.html
do_default=1    # refresh rag_db        (CVL only; used by AI Composer + cex-analyzer)
do_extended=0   # refresh extended_rag_db (CVL + prover + user-guide; used by sanity-analyzer)

usage() {
    cat <<'EOF'
Usage: scripts/refresh_rag.sh [options]

Regenerate the documentation and rebuild the CVL-manual RAG from scratch.

Options:
  --skip-gen-docs   Skip the gen_docs.sh step and rebuild from the HTML already
                    present in scripts/prover-docs/. Use when the docs are fresh.
  --extended        ALSO refresh extended_rag_db (CVL + prover + user-guide).
                    Requires that DB / user to already be provisioned.
  --all             Refresh both rag_db and extended_rag_db.
  --only-extended   Refresh ONLY extended_rag_db (skip rag_db).
  -h, --help        Show this help.

Default (no options): regenerate docs, then wipe+rebuild rag_db only.

Examples:
  scripts/refresh_rag.sh                 # docs -> wipe+rebuild rag_db
  scripts/refresh_rag.sh --all           # docs -> wipe+rebuild both databases
  scripts/refresh_rag.sh --skip-gen-docs # rebuild rag_db from existing HTML
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-gen-docs) do_gen=0 ;;
        --extended)      do_extended=1 ;;
        --all)           do_default=1; do_extended=1 ;;
        --only-extended) do_default=0; do_extended=1 ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; echo >&2; usage >&2; exit 1 ;;
    esac
    shift
done

# Read wipe_rag.py's confirmation sentinel from source so this never drifts if
# the literal changes. If wipe_rag.py rejects what we pipe, it exits non-zero and
# `set -e` aborts before any rebuild — failing safe (nothing wiped, nothing half-built).
confirmation="$(grep -E '^_CONFIRMATION[[:space:]]*=' "$script_dir/wipe_rag.py" | sed -E 's/.*"(.*)".*/\1/')"
if [[ -z "$confirmation" ]]; then
    echo "Error: could not read the confirmation sentinel from $script_dir/wipe_rag.py" >&2
    exit 1
fi

# wipe a RAG database by piping the auto-confirmation into wipe_rag.py.
# $1 = human label for logging; remaining args are passed through to wipe_rag.py.
wipe_db() {
    local label="$1"; shift
    echo "==> Wiping ${label} ..."
    printf '%s\n' "$confirmation" | ( cd "$parent" && uv run python "$script_dir/wipe_rag.py" "$@" )
}

echo "============================================================"
echo "  RAG refresh"
echo "    regenerate docs : $([[ $do_gen -eq 1 ]] && echo yes || echo 'no (reuse prover-docs/)')"
echo "    rag_db          : $([[ $do_default -eq 1 ]] && echo 'wipe + rebuild' || echo skip)"
echo "    extended_rag_db : $([[ $do_extended -eq 1 ]] && echo 'wipe + rebuild' || echo skip)"
echo "  This empties the target database(s) then rebuilds. Run offline."
echo "============================================================"

# 1. (re)generate documentation HTML into scripts/prover-docs/
if [[ $do_gen -eq 1 ]]; then
    echo "==> Regenerating documentation HTML (gen_docs.sh) ..."
    bash "$script_dir/gen_docs.sh"
fi

# 2+3. rag_db (CVL only)
if [[ $do_default -eq 1 ]]; then
    wipe_db "rag_db (CVL only)"
    echo "==> Rebuilding rag_db (populate_rag.sh) ..."
    bash "$script_dir/populate_rag.sh"
fi

# 2+3. extended_rag_db (CVL + prover + user-guide)
if [[ $do_extended -eq 1 ]]; then
    wipe_db "extended_rag_db (CVL + prover + user-guide)" --conn-string "$ext_conn"
    echo "==> Rebuilding extended_rag_db (populate_extended_rag.sh) ..."
    bash "$script_dir/populate_extended_rag.sh"
fi

echo
echo "✅ RAG refresh complete."
