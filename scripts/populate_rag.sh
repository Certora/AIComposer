#!/bin/bash
set -euo pipefail

script_dir="$(realpath "$(dirname "$0")")"

docs_dir="$script_dir/prover-docs"
if [[ ! -f "$docs_dir/cvl.html" ]]; then
    echo "Error: $docs_dir/cvl.html not found. Run ./gen_docs.sh first." >&2
    exit 1
fi

parent=$(realpath "$script_dir/..")

(cd $parent; uv run --isolated --group ragbuild -s $script_dir/ragbuild.py "$docs_dir/cvl.html")
