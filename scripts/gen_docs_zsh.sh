#!/bin/zsh
set -euo pipefail

host_dir="$(realpath "$(dirname "$0")")/prover-docs"
mkdir -p "$host_dir"
doc_dir="$(mktemp -d)"
venv_dir="$(mktemp -d)"

cleanup() {
    [[ $(type -t deactivate) == function ]] && deactivate
    rm -rf "$doc_dir" "$venv_dir"
}
trap cleanup EXIT

git clone --depth 1 git@github.com:Certora/Documentation.git "$doc_dir"

python3 -m venv "$venv_dir"
source "$venv_dir/bin/activate"
pip install -r "$doc_dir/requirements.txt"

for target in "$doc_dir"/docs/{cvl,solana,prover}/ ; do
    cp "$doc_dir/conf.py" "$target"
    cd "$target"
    sphinx-build -M singlehtml . tmp
    cp tmp/singlehtml/index.html "$host_dir/$(basename "$target").html"
done
