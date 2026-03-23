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
for f in cvl.html prover.html user-guide.html; do
    if [[ ! -f "$docs_dir/$f" ]]; then
        echo "Error: $docs_dir/$f not found. Run ./gen_docs.sh first." >&2
        exit 1
    fi
done

python3 -m venv "$venv_dir"
source "$venv_dir/bin/activate"
pip install -r "$script_dir/rag_build_requirements.txt"

python3 "$script_dir/ragbuild.py" \
    --connection "postgresql://extended_rag_user:rag_password@localhost:5432/extended_rag_db" \
    "$docs_dir/cvl.html" \
    "$docs_dir/prover.html" \
    "$docs_dir/user-guide.html"
