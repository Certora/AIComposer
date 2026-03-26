#!/bin/bash
set -euo pipefail

script_dir="$(realpath "$(dirname "$0")")"
venv_dir="$(mktemp -d)"

cleanup() {
    [[ $(type -t deactivate) == function ]] && deactivate
    rm -rf "$venv_dir"
}
trap cleanup EXIT

docs_dir="$script_dir/prover-docs"
if [[ ! -f "$docs_dir/cvl.html" ]]; then
    echo "Error: $docs_dir/cvl.html not found. Run ./gen_docs.sh first." >&2
    exit 1
fi

python3 -m venv "$venv_dir"
source "$venv_dir/bin/activate"
pip install -r "$script_dir/rag_build_requirements.txt"

python3 "$script_dir/ragbuild.py" "$docs_dir/cvl.html"
