#!/usr/bin/env python3

"""
Simple script to build extended manual by combining existing CVL manual 
with selected prover documentation markdown files.
"""

import pathlib
import json
import subprocess
from bs4 import BeautifulSoup
import markdown

# Selected prover documentation files (relative to temp_docs/docs/)
PROVER_FILES = [
    "prover/cli/options.md",              # dynamic_bound, dynamic_dispatch
    "prover/cli/conf-file-api.md",        # configuration
    "prover/checking/sanity.md",          # sanity checks, vacuity
    "prover/checking/index.md",           # checking process
    "prover/checking/coverage-info.md",   # coverage analysis
    "prover/approx/loops.md",             # loop_iter, loop unrolling
    "prover/approx/hashing.md",           # hashing_length_bound
    "prover/approx/index.md",             # approximation overview
    "prover/diagnosis/index.md",          # debugging failures
    "user-guide/checking.md",             # checking guide
    "user-guide/gaps.md",                 # gaps and limitations
    "user-guide/glossary.md",             # glossary
    "user-guide/multicontract/index.md",  # multicontract usage
]

def main():
    script_dir = pathlib.Path(__file__).parent
    project_root = script_dir.parent
    
    print("🔨 Building extended manual...")
    
    # Check if CVL manual exists
    cvl_manual = script_dir / "cvl_manual.html"
    if not cvl_manual.exists():
        print("❌ cvl_manual.html not found. Run gen_docs.sh first.")
        return 1
    
    # Check if temp_docs exists
    temp_docs = project_root / "temp_docs"
    if not temp_docs.exists():
        print("📥 Cloning documentation repository...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/Certora/Documentation.git",
            str(temp_docs)
        ], check=True)
    
    print("📖 Loading CVL manual...")
    with open(cvl_manual, 'r', encoding='utf-8') as f:
        cvl_content = f.read()
    
    cvl_soup = BeautifulSoup(cvl_content, 'html.parser')
    
    print("📄 Processing prover documentation files...")
    added_sections = 0
    
    # Add separator and header for prover docs
    if cvl_soup.body:
        separator = cvl_soup.new_tag('hr', style='margin-top: 3em; margin-bottom: 3em;')
        cvl_soup.body.append(separator)
        
        header = cvl_soup.new_tag('h1', id='prover-documentation')
        header.string = '📋 Additional Prover Documentation'
        cvl_soup.body.append(header)
        
        subtitle = cvl_soup.new_tag('p')
        subtitle.string = 'The following sections contain essential prover configuration and debugging information.'
        cvl_soup.body.append(subtitle)
    
    for file_path in PROVER_FILES:
        full_path = temp_docs / "docs" / file_path
        if full_path.exists():
            print(f"   ➕ Adding: {file_path}")
            
            # Read markdown
            with open(full_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # Convert to HTML
            html_content = markdown.markdown(
                md_content, 
                extensions=['fenced_code', 'tables', 'codehilite', 'toc']
            )
            
            # Create section
            section_title = file_path.replace('/', ' → ').replace('.md', '').replace('index', 'Overview')
            section_div = cvl_soup.new_tag('div', **{'class': 'prover-section', 'id': file_path.replace('/', '-').replace('.md', '')})
            
            # Add section header
            section_header = cvl_soup.new_tag('h2')
            section_header.string = f"📘 {section_title}"
            section_div.append(section_header)
            
            # Add content
            content_soup = BeautifulSoup(html_content, 'html.parser')
            for element in content_soup.find_all():
                section_div.append(element)
            
            cvl_soup.body.append(section_div)
            added_sections += 1
    
    # Update title
    if cvl_soup.title:
        cvl_soup.title.string = 'Extended Certora Documentation (CVL + Prover)'
    
    # Write extended manual
    extended_manual = script_dir / "extended_manual.html"
    print(f"💾 Writing extended manual: {extended_manual}")
    
    with open(extended_manual, 'w', encoding='utf-8') as f:
        f.write(str(cvl_soup))
    
    print(f"✅ Extended manual created successfully!")
    print(f"   📚 Original CVL sections: preserved")
    print(f"   🔧 Added prover sections: {added_sections}")
    print(f"   📄 Output: {extended_manual}")
    
    return 0

if __name__ == "__main__":
    exit(main())