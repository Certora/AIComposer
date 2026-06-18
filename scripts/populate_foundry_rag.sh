#!/bin/bash
# End-to-end builder for the foundry cheatcode RAG.
#
#   1. clone the foundry book repo (shallow)
#   2. convert the cheatcode .mdx pages to .html (composer.scripts.foundry_process)
#   3. ingest the .html into the RAG db (composer.scripts.foundry_ragbuild)
#   4. clean up the scratch clone + generated html on exit
#
# Any extra args are forwarded to foundry_ragbuild, e.g.:
#   ./populate_foundry_rag.sh --print          # dry-run, print chunks
#   ./populate_foundry_rag.sh --output <conn>  # write to a specific db
set -euo pipefail

script_dir="$(realpath "$(dirname "$0")")"
parent="$(realpath "$script_dir/..")"

BOOK_REPO="https://github.com/foundry-rs/book/"
# Cheatcode reference pages (one .mdx per cheatcode) within the book repo.
CHEATCODES_SUBPATH="src/pages/reference/cheatcodes"

# Scratch area for the clone + generated HTML; removed on exit (success or fail).
workdir="$(mktemp -d)"
cleanup() { rm -rf "$workdir"; }
trap cleanup EXIT

echo "Cloning $BOOK_REPO (shallow) ..." >&2
git clone --depth 1 "$BOOK_REPO" "$workdir/book"

mdx_dir="$workdir/book/$CHEATCODES_SUBPATH"
if [[ ! -d "$mdx_dir" ]]; then
    echo "Error: cheatcode docs not found at '$CHEATCODES_SUBPATH' in the cloned book." >&2
    echo "The book layout may have moved; update CHEATCODES_SUBPATH in this script." >&2
    exit 1
fi

html_dir="$workdir/html"
echo "Converting .mdx -> .html ..." >&2
(cd "$parent"; uv run --isolated --group ragbuild \
    python -m composer.scripts.foundry_process "$mdx_dir" "$html_dir")

shopt -s globstar nullglob
html_files=("$html_dir"/**/*.html)
if [[ ${#html_files[@]} -eq 0 ]]; then
    echo "Error: no HTML produced from $mdx_dir" >&2
    exit 1
fi

echo "Ingesting ${#html_files[@]} cheatcode page(s) into the RAG database ..." >&2
(cd "$parent"; uv run --isolated --group ragbuild \
    python -m composer.scripts.foundry_ragbuild "${html_files[@]}" "$@")

echo "Done." >&2
