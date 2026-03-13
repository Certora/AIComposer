#!/bin/bash
set -euo pipefail

# Extended version of gen_docs.sh that builds:
# 1. Existing CVL documentation (unchanged)
# 2. Selected prover documentation pages (multi-page html for use with rag_config_extended.json)

host_dir=$(realpath $(dirname $0))
doc_dir=$(mktemp -d)
venv_dir=$(mktemp -d)
do_pop=0

cleanup () {
    deactivate 2>/dev/null || true
    rm -rf $doc_dir $venv_dir
    if [[ $do_pop -eq 1 ]]; then
        popd
    fi
}
trap cleanup EXIT

echo "Building extended documentation (CVL + selected prover pages)..."

# Clone and setup
git clone --depth 1 git@github.com:Certora/Documentation.git $doc_dir
pushd $doc_dir
do_pop=1
python3 -m venv $venv_dir
source $venv_dir/bin/activate
pip install -r requirements.txt

# Build existing CVL documentation (unchanged)
echo "Building CVL documentation..."
cp ./conf.py ./docs/cvl/
cd docs/cvl/
sphinx-build -M singlehtml . tmp
cp tmp/singlehtml/index.html $host_dir/cvl_manual.html

# Build prover documentation (multi-page html for individual page access)
echo "Building prover documentation..."
cd $doc_dir
sphinx-build -M html . tmp_html
cp -r tmp_html/html $host_dir/prover_html

echo "Documentation build complete!"
echo "- CVL manual: cvl_manual.html"
echo "- Prover HTML pages: prover_html/"
echo "- Run: python3 ragbuild.py --config rag_config_extended.json --connection <extended_rag_db>"
