#!/bin/bash

# Extended version of gen_docs.sh that builds:
# 1. Existing CVL documentation (unchanged)
# 2. Selected prover documentation pages
# 3. Combined extended manual

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

# Build selected prover pages
echo "Building additional prover documentation..."
cd ../

# Create a minimal build with just the selected prover pages
mkdir -p extended_build/prover/approx
mkdir -p extended_build/prover/checking  
mkdir -p extended_build/prover/cli
mkdir -p extended_build/prover/diagnosis
mkdir -p extended_build/user-guide

# Copy selected files
cp prover/approx/hashing.md extended_build/prover/approx/
cp prover/approx/loops.md extended_build/prover/approx/
cp prover/approx/index.md extended_build/prover/approx/
cp prover/checking/sanity.md extended_build/prover/checking/
cp prover/checking/index.md extended_build/prover/checking/
cp prover/checking/coverage-info.md extended_build/prover/checking/
cp prover/cli/options.md extended_build/prover/cli/
cp prover/cli/conf-file-api.md extended_build/prover/cli/
cp prover/diagnosis/index.md extended_build/prover/diagnosis/
cp user-guide/checking.md extended_build/user-guide/
cp user-guide/gaps.md extended_build/user-guide/
cp user-guide/glossary.md extended_build/user-guide/
cp user-guide/multicontract/index.md extended_build/user-guide/

# Create a simple index for prover docs
cat > extended_build/index.rst << 'EOF'
Additional Prover Documentation
==============================

.. toctree::
   :maxdepth: 2
   :caption: Prover CLI
   
   prover/cli/options
   prover/cli/conf-file-api

.. toctree::
   :maxdepth: 2
   :caption: Verification Process
   
   prover/checking/index
   prover/checking/sanity
   prover/checking/coverage-info

.. toctree::
   :maxdepth: 2
   :caption: Approximation Techniques
   
   prover/approx/index
   prover/approx/loops
   prover/approx/hashing

.. toctree::
   :maxdepth: 2
   :caption: Debugging
   
   prover/diagnosis/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   user-guide/checking
   user-guide/gaps
   user-guide/glossary
   user-guide/multicontract/index
EOF

# Copy configuration
cp ../conf.py extended_build/

# Build prover documentation
cd extended_build
sphinx-build -M singlehtml . tmp_prover
cd ..

# Combine CVL and prover documentation
python3 -c "
import sys
from bs4 import BeautifulSoup

# Read CVL manual
with open('$host_dir/cvl_manual.html', 'r') as f:
    cvl_content = f.read()

# Read prover manual  
with open('extended_build/tmp_prover/singlehtml/index.html', 'r') as f:
    prover_content = f.read()

# Parse both
cvl_soup = BeautifulSoup(cvl_content, 'html.parser')
prover_soup = BeautifulSoup(prover_content, 'html.parser')

# Add prover content to CVL body
if cvl_soup.body and prover_soup.body:
    # Add separator
    separator = cvl_soup.new_tag('hr')
    cvl_soup.body.append(separator)
    
    header = cvl_soup.new_tag('h1')
    header.string = 'Additional Prover Documentation'
    cvl_soup.body.append(header)
    
    # Append prover content
    for element in prover_soup.body.find_all(recursive=False):
        cvl_soup.body.append(element)

# Update title
if cvl_soup.title:
    cvl_soup.title.string = 'Extended Certora Documentation (CVL + Prover)'

# Write combined manual
with open('$host_dir/extended_manual.html', 'w') as f:
    f.write(str(cvl_soup))

print('✅ Extended manual created successfully!')
"

echo "Documentation build complete!"
echo "- CVL manual: cvl_manual.html (backward compatibility)"  
echo "- Extended manual: extended_manual.html (includes selected prover docs)"