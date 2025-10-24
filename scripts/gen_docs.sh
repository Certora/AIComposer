#!/bin/bash

host_dir=$(realpath $(dirname $0))
doc_dir=$(mktemp -d)
venv_dir=$(mktemp -d)
do_pop=0
cleanup () {
    deactivate
    rm -rf $doc_dir $venv_dir
    if [[ $do_pop -eq 1 ]]; then
        popd
    fi

}
trap cleanup EXIT
git clone --depth 1 git@github.com:Certora/Documentation.git $doc_dir
pushd $doc_dir
do_pop=1
python3 -m venv $venv_dir
source $venv_dir/bin/activate
pip install -r requirements.txt
cp ./conf.py ./docs/cvl/
cd docs/cvl/
sphinx-build -M singlehtml . tmp
cp tmp/singlehtml/index.html $host_dir/cvl_manual.html