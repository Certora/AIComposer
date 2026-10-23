# **COMPLETE CERTORA CEX EXTRACTION & CLAUDE ANALYSIS SETUP GUIDE**

**Author's Note:** This is a PhD-level, production-grade setup. You configure it once and use it across ALL projects indefinitely.

---

## **Table of Contents**

1. [Architecture Overview](#architecture-overview)
2. [One-Time Global Setup](#one-time-global-setup)
3. [Per-Project Minimal Setup](#per-project-minimal-setup)
4. [Deep Dive: Understanding Each Component](#deep-dive)
5. [Integration with Your Workflow](#workflow-integration)
6. [Advanced Configurations](#advanced-configurations)

---

# **ARCHITECTURE OVERVIEW**

## **How It Works**

```
┌─────────────────────────────────────────────────────────────────┐
│ YOUR DEVELOPMENT WORKFLOW                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Project A    Project B    Project C    Project D              │
│  (DeFi)       (NFT)        (DAO)        (Bridge)               │
│     │             │           │           │                    │
│     └─────────────┴───────────┴───────────┘                    │
│                   ↓                                             │
│      GLOBAL CERTORA TOOLS (Set Up Once)                         │
│      ├── Python Environment (venv, deps)                        │
│      ├── Claude API Key                                         │
│      ├── extract_and_analyze_cex.py                             │
│      ├── Configuration profiles                                 │
│      └── Shell aliases for common commands                      │
│                   ↓                                             │
│      RE-USED ACROSS ALL PROJECTS                                │
│                   ↓                                             │
│      ✅ ONE SETUP = ALL PROJECTS AUTOMATED                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# **ONE-TIME GLOBAL SETUP**

This happens **once on your machine**. After this, you never do it again.

## **Step 1: Create Global Certora Tools Directory**

```bash
# Create a central location for all Certora tools
mkdir -p ~/certora-tools
cd ~/certora-tools

# This will be your "command center" for all projects
```

## **Step 2: Set Up Python Virtual Environment**

```bash
# Create isolated Python environment (best practice)
python3 -m venv venv

# Activate it
source venv/bin/activate

# Verify Python
python --version  # Should be 3.9+
pip --version
```

## **Step 3: Install Dependencies**

```bash
# Install Claude API client and utilities
pip install anthropic python-dotenv

# Verify installation
python -c "import anthropic; print('✓ Anthropic installed')"
```

## **Step 4: Download the Extraction Tool**

```bash
# Copy the extraction tool to this central location
# Option A: Download from repo
curl -o extract_and_analyze_cex.py \
  https://raw.githubusercontent.com/0xbrett8571/CertoraAIComposer/master/extract_and_analyze_cex.py

# Option B: Copy from AIComposer if you cloned it
cp /path/to/CertoraAIComposer/extract_and_analyze_cex.py .

# Verify it works
python extract_and_analyze_cex.py --help
```

## **Step 5: Create Configuration File**

Create `~/.certora-config`:

```bash
# This file stores your configuration
cat > ~/.certora-config << 'EOF'
# Certora Tools Configuration
# Source this file to set up environment

# Claude API Configuration
export ANTHROPIC_API_KEY="sk-ant-YOUR-KEY-HERE"

# Python Environment
CERTORA_TOOLS_HOME="$HOME/certora-tools"
export PYTHON_ENV="$CERTORA_TOOLS_HOME/venv"

# Tool Paths
export CEX_TOOL="$CERTORA_TOOLS_HOME/extract_and_analyze_cex.py"
export CERTORA_TOOL_BIN="$PYTHON_ENV/bin/python"

# Default settings
export CERTORA_OUTPUT_DIR="/tmp/certoraOutput"
export CERTORA_SPEC_FORMAT="cvl"  # or "sol" for Solidity specs
export CLAUDE_MODEL="claude-3-5-sonnet-20241022"

# Logging
export CERTORA_LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
EOF

chmod 600 ~/.certora-config  # Keep it private (contains API key)
```

## **Step 6: Add to Your Shell Profile**

Add this to `~/.bashrc` or `~/.zshrc`:

```bash
# Load Certora tools (at end of file)
if [ -f "$HOME/.certora-config" ]; then
    source "$HOME/.certora-config"
fi

# Convenience aliases
alias cex-tools='cd $CERTORA_TOOLS_HOME && source venv/bin/activate'
alias cex-list='$CERTORA_TOOL_BIN $CEX_TOOL list'
alias cex-extract='$CERTORA_TOOL_BIN $CEX_TOOL extract'
alias cex-analyze='$CERTORA_TOOL_BIN $CEX_TOOL analyze'
```

## **Step 7: Set Your Claude API Key**

```bash
# Edit the config file with your actual API key
# Option A: Edit directly
nano ~/.certora-config
# Find: export ANTHROPIC_API_KEY="sk-ant-YOUR-KEY-HERE"
# Replace with your actual key

# Option B: Update via command
sed -i 's/sk-ant-YOUR-KEY-HERE/sk-ant-YOUR-ACTUAL-KEY/' ~/.certora-config

# Verify it's set
echo $ANTHROPIC_API_KEY  # Should not be empty
```

## **Step 8: Verify Global Setup**

```bash
# Activate tools
source ~/.certora-config

# Test everything
echo "✓ Certora Tools Home: $CERTORA_TOOLS_HOME"
echo "✓ Python Environment: $PYTHON_ENV"
echo "✓ CEX Tool: $CEX_TOOL"
echo "✓ API Key set: $([ -z "$ANTHROPIC_API_KEY" ] && echo "❌ NO" || echo "✓ YES")"

# Test tool
$CERTORA_TOOL_BIN $CEX_TOOL --help | head -20
```

**Expected Output:**
```
✓ Certora Tools Home: /Users/yourname/certora-tools
✓ Python Environment: /Users/yourname/certora-tools/venv
✓ CEX Tool: /Users/yourname/certora-tools/extract_and_analyze_cex.py
✓ API Key set: ✓ YES

usage: extract_and_analyze_cex.py [-h] {list,extract,analyze} ...

Certora Prover CEX Extraction & Analysis Tool
...
```

---

# **PER-PROJECT MINIMAL SETUP**

After global setup, each project needs **almost nothing**.

## **Project Structure**

```bash
# Your project directory
my-defi-protocol/
├── contracts/
│   ├── Token.sol
│   ├── Vault.sol
│   └── Treasury.sol
├── spec/
│   ├── Token.spec
│   ├── Vault.spec
│   └── Treasury.spec
├── config.conf              ← Certora config
├── prover_results/          ← Output (generated)
└── .certora-project         ← Project config (NEW)
```

## **Step 1: Create Project Marker File**

In each project root, create `.certora-project`:

```bash
# Navigate to your project
cd my-defi-protocol

# Create marker file
cat > .certora-project << 'EOF'
# Certora Project Configuration
# This file marks the project root for Certora tools

PROJECT_NAME="my-defi-protocol"
CERTORA_OUTPUT_DIR="./prover_results"
SPEC_DIR="./spec"
CONTRACT_DIR="./contracts"
MAIN_CONTRACT="Token"

# Custom settings for this project (optional)
# CLAUDE_MODEL="claude-3-opus-20250219"  # Override if needed
# LOOP_ITER=5
EOF
```

## **Step 2: Create Shortcut Script (Optional)**

In your project root, create `scripts/analyze_cex.sh`:

```bash
#!/bin/bash
# Quick access to CEX analysis in this project

# Load global config
source ~/.certora-config

# Load project config if it exists
if [ -f ./.certora-project ]; then
    source ./.certora-project
fi

# Use project-specific output dir
OUTPUT_DIR="${CERTORA_OUTPUT_DIR:-./ prover_results}"

# Run the tool
$CERTORA_TOOL_BIN $CEX_TOOL "$@" "$OUTPUT_DIR"
EOF

chmod +x scripts/analyze_cex.sh
```

## **Step 3: Add to .gitignore**

```bash
# Don't commit analysis files (they're regeneratable)
cat >> .gitignore << 'EOF'

# Certora Prover outputs
prover_results/
*.cex.txt
*_analysis.md
*.certora-cache

# Temporary files
/tmp/
EOF
```

---

# **DEEP DIVE: Understanding Each Component**

## **Component 1: Global Configuration (~/.certora-config)**

### **What It Does:**
- Sets environment variables that are used by ALL projects
- Stores your Claude API key securely
- Defines default tool locations

### **Key Variables:**

| Variable | Purpose | Example |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | Claude authentication | `sk-ant-...` |
| `CERTORA_TOOLS_HOME` | Central tools location | `/home/user/certora-tools` |
| `CEX_TOOL` | Path to extraction tool | `$CERTORA_TOOLS_HOME/extract_and_analyze_cex.py` |
| `CLAUDE_MODEL` | Which Claude to use | `claude-3-5-sonnet-20241022` |
| `CERTORA_OUTPUT_DIR` | Default output location | `/tmp/certoraOutput` |

### **Security Best Practice:**

```bash
# Keep API key in .certora-config private
chmod 600 ~/.certora-config

# Verify permissions
ls -la ~/.certora-config
# Should show: -rw------- (owner read-write only)
```

---

## **Component 2: Python Virtual Environment**

### **Why It's Needed:**

The virtual environment isolates Certora tools from your system Python, preventing:
- Dependency conflicts
- System-wide pollution
- Version mismatches

### **Managing the Environment:**

```bash
# Activate when you need it
source ~/certora-tools/venv/bin/activate

# Deactivate when done
deactivate

# Check what's installed
pip list

# Update dependencies
pip install --upgrade anthropic python-dotenv
```

---

## **Component 3: The Extraction Tool (extract_and_analyze_cex.py)**

### **What It Does:**

```python
# High-level workflow
1. Reads Certora JSON output structure
2. Parses treeViewStatus_*.json files
3. Extracts counterexample data
4. Formats for human/Claude consumption
5. Sends to Claude API for analysis
6. Saves results to markdown
```

### **Key Classes:**

```
CertoraResultsParser
├── Finds latest treeview file
├── Loads JSON
└── Searches for rules

CEXExtractor
├── Locates CEX files
├── Extracts JSON data
└── Formats as tree

ClaudeAnalyzer
├── Creates prompts
├── Calls Claude API
└── Returns analysis

CLIFormatter
├── Formats output
├── Generates markdown
└── Pretty-prints
```

---

## **Component 4: Per-Project Configuration (.certora-project)**

### **Customization per Project:**

```bash
# Example: High-security DeFi project (lots of rules)
PROJECT_NAME="defi-protocol"
CERTORA_OUTPUT_DIR="./prover_results"
SPEC_DIR="./spec"
MAIN_CONTRACT="Treasury"
LOOP_ITER=6              # More iterations for complex logic
HASHING_BOUND=256        # Larger hashing bound for security
CLAUDE_MODEL="claude-3-opus-20250219"  # More capable model
```

```bash
# Example: Simple NFT project (fewer rules)
PROJECT_NAME="nft-contract"
CERTORA_OUTPUT_DIR="./prover_results"
SPEC_DIR="./spec"
MAIN_CONTRACT="NFTCollection"
LOOP_ITER=3              # Standard iterations
CLAUDE_MODEL="claude-3-5-sonnet-20241022"  # Standard model
```

---

# **WORKFLOW INTEGRATION**

## **Typical Day: How You'd Use This**

### **Morning: Run Prover on Your DeFi Project**

```bash
cd ~/projects/my-defi-protocol

# Load global config (happens automatically if in .bashrc)
source ~/.certora-config

# Run Certora (results download to ./prover_results)
certoraRun config.conf --output_dir ./prover_results
# Takes 5 minutes to 2 hours depending on complexity

# Meanwhile, grab coffee ☕
```

### **Mid-Morning: List Failed Rules**

```bash
# Quick check what failed
cex-list ./prover_results

# Output:
# Available Rules in Treeview:
# 
# Rule Name                    Status          CEX Available  
# ---------------------------------------------------------------
# transfer_succeeds            VIOLATED        ✓ Yes          
# balanceInvariant             VERIFIED        ✗ No           
# noNegativeBalance            VIOLATED        ✓ Yes          
```

### **Late Morning: Analyze First Failure**

```bash
# Get Claude's analysis
cex-analyze ./prover_results transfer_succeeds \
  --spec ./spec/Token.spec \
  --output analysis_transfer.md

# Read the analysis
cat analysis_transfer.md

# Output shows:
# - What scenario caused the failure
# - Root cause in your code
# - 2-3 specific fixes
```

### **Noon: Apply Fixes**

```bash
# Edit based on Claude's suggestions
vim contracts/Token.sol

# Add the fix (e.g., missing require statement)
```

### **Afternoon: Re-run Prover to Verify**

```bash
certoraRun config.conf --output_dir ./prover_results_v2

# List rules again
cex-list ./prover_results_v2

# Check if transfer_succeeds is now VERIFIED
```

### **End of Day: Document Progress**

```bash
# Analyze remaining failures
cex-analyze ./prover_results_v2 noNegativeBalance \
  --spec ./spec/Token.spec \
  --output analysis_noNegativeBalance.md

# Commit progress
git add contracts/Token.sol analysis_*.md
git commit -m "Fix transfer_succeeds rule violation"
```

---

# **ADVANCED CONFIGURATIONS**

## **Configuration 1: Batch Automation Script**

Create `scripts/analyze_all_failures.sh`:

```bash
#!/bin/bash
# Analyze all failed rules in parallel

source ~/.certora-config

if [ -f ./.certora-project ]; then
    source ./.certora-project
fi

OUTPUT_DIR="${CERTORA_OUTPUT_DIR:-./ prover_results}"
SPEC_FILE="${SPEC_DIR:-./ spec}/*.spec"

echo "🔍 Finding all failed rules..."

# Get all VIOLATED rules
FAILED_RULES=$($CERTORA_TOOL_BIN $CEX_TOOL list "$OUTPUT_DIR" | \
    grep VIOLATED | awk '{print $1}')

if [ -z "$FAILED_RULES" ]; then
    echo "✅ All rules passing! 🎉"
    exit 0
fi

TOTAL=$(echo "$FAILED_RULES" | wc -l)
echo "📊 Found $TOTAL failed rules. Analyzing..."
echo ""

# Analyze each rule
COUNT=0
for RULE in $FAILED_RULES; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] 📊 Analyzing: $RULE"
    
    $CERTORA_TOOL_BIN $CEX_TOOL analyze "$OUTPUT_DIR" "$RULE" \
      --spec "$(ls $SPEC_FILE | head -1)" \
      --output "analysis_${RULE}.md"
    
    echo "✓ Saved: analysis_${RULE}.md"
done

echo ""
echo "✅ All $TOTAL rules analyzed!"
echo ""
echo "📋 Summary:"
ls -lh analysis_*.md | awk '{print "  " $9 " (" $5 ")"}'
```

**Usage:**

```bash
chmod +x scripts/analyze_all_failures.sh
./scripts/analyze_all_failures.sh
```

---

## **Configuration 2: Continuous Verification Loop**

Create `scripts/verify_loop.sh`:

```bash
#!/bin/bash
# Continuously run prover, fix failures, repeat

source ~/.certora-config

if [ -f ./.certora-project ]; then
    source ./.certora-project
fi

ITERATION=0
MAX_ITERATIONS=10

while [ $ITERATION -lt $MAX_ITERATIONS ]; do
    ITERATION=$((ITERATION + 1))
    
    echo "═══════════════════════════════════════════════════════"
    echo "Iteration $ITERATION/$MAX_ITERATIONS"
    echo "═══════════════════════════════════════════════════════"
    
    # Run prover
    echo "🚀 Running Certora Prover..."
    certoraRun config.conf --output_dir ./prover_results_iter$ITERATION
    
    # Check results
    FAILED=$($CERTORA_TOOL_BIN $CEX_TOOL list "./prover_results_iter$ITERATION" | \
        grep VIOLATED | wc -l)
    
    if [ $FAILED -eq 0 ]; then
        echo "✅ All rules VERIFIED! 🎉"
        echo "Verification completed in $ITERATION iterations"
        exit 0
    fi
    
    echo "❌ $FAILED rules still failing"
    
    # Analyze first failure
    FIRST_FAILED=$($CERTORA_TOOL_BIN $CEX_TOOL list "./prover_results_iter$ITERATION" | \
        grep VIOLATED | head -1 | awk '{print $1}')
    
    echo ""
    echo "📊 Analyzing first failure: $FIRST_FAILED"
    
    $CERTORA_TOOL_BIN $CEX_TOOL analyze "./prover_results_iter$ITERATION" "$FIRST_FAILED" \
      --spec ./spec/$(ls spec/ | head -1) \
      --output "iteration_${ITERATION}_analysis.md"
    
    echo ""
    echo "📖 Review the analysis:"
    cat "iteration_${ITERATION}_analysis.md"
    
    echo ""
    read -p "Press ENTER after fixing the code..."
done

echo "⚠️  Reached maximum iterations without full verification"
```

---

## **Configuration 3: CI/CD Integration**

Create `.github/workflows/verify.yml`:

```yaml
name: Formal Verification

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install anthropic python-dotenv
      
      - name: Download CEX tool
        run: |
          curl -o extract_and_analyze_cex.py \
            https://raw.githubusercontent.com/0xbrett8571/CertoraAIComposer/master/extract_and_analyze_cex.py
      
      - name: Run Certora Prover
        run: |
          certoraRun config.conf --output_dir ./prover_results
      
      - name: Check verification results
        run: |
          FAILED=$(python extract_and_analyze_cex.py list ./prover_results | grep VIOLATED | wc -l)
          if [ $FAILED -gt 0 ]; then
            echo "❌ $FAILED rules failed verification"
            exit 1
          fi
          echo "✅ All rules verified"
      
      - name: Generate analysis for failed rules
        if: failure()
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          python extract_and_analyze_cex.py list ./prover_results | grep VIOLATED | \
            awk '{print $1}' | while read rule; do
              python extract_and_analyze_cex.py analyze ./prover_results "$rule" \
                --spec ./spec/main.spec \
                --output "analysis_${rule}.md"
          done
      
      - name: Upload analysis artifacts
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: verification-analysis
          path: analysis_*.md
```

---

# **TROUBLESHOOTING**

## **Issue 1: "ANTHROPIC_API_KEY not set"**

```bash
# Verify configuration is loaded
echo $ANTHROPIC_API_KEY

# If empty:
source ~/.certora-config
echo $ANTHROPIC_API_KEY  # Should show something

# If still empty, check file
cat ~/.certora-config | grep ANTHROPIC_API_KEY
# Should show: export ANTHROPIC_API_KEY="sk-ant-..."
```

## **Issue 2: "Python package not found"**

```bash
# Ensure venv is activated
source ~/certora-tools/venv/bin/activate

# Check pip list
pip list | grep anthropic  # Should show installed package

# If not found, reinstall
pip install anthropic
```

## **Issue 3: "CEX Tool not found"**

```bash
# Verify file exists
ls -la $CEX_TOOL

# If it doesn't exist, download it
curl -o $CEX_TOOL \
  https://raw.githubusercontent.com/0xbrett8571/CertoraAIComposer/master/extract_and_analyze_cex.py

# Make executable
chmod +x $CEX_TOOL
```

## **Issue 4: "Permission denied" on analyze.sh**

```bash
# Make script executable
chmod +x scripts/analyze_cex.sh
chmod +x scripts/analyze_all_failures.sh

# Verify
ls -la scripts/*.sh  # Should show -rwxr-xr-x
```

---

# **FINAL SUMMARY**

## **What You Configure ONCE (Global)**

✅ Python virtual environment  
✅ Claude API key  
✅ Global configuration file (~/.certora-config)  
✅ Shell aliases and functions  
✅ Download CEX tool  

**Time: ~15 minutes**

---

## **What Each Project Needs (Minimal)**

✅ `.certora-project` marker file  
✅ `config.conf` for Certora settings  
✅ Optional: `scripts/analyze_cex.sh` shortcut  

**Time: ~2 minutes per project**

---

## **Reuse Across Projects**

Once global setup is done, you can run this on ANY project:

```bash
# Works on any project with config.conf
certoraRun config.conf --output_dir ./prover_results

# Extract and analyze
cex-list ./prover_results
cex-analyze ./prover_results rule_name --spec ./spec/file.spec
```

**Same 4 commands work everywhere!**

---

## **Advanced: What Professionals Do**

1. Global setup in `~/certora-tools/`
2. Central Python environment with all tools
3. Project-specific marker files (`.certora-project`)
4. Automated analysis scripts in `scripts/` folder
5. CI/CD integration for automatic verification
6. Git-tracked analysis for history

**This is production-grade infrastructure.** Use it!

---

**Next Steps:**

1. Follow the "One-Time Global Setup" (15 min)
2. Create `.certora-project` in your project (2 min)
3. Run your first: `certoraRun config.conf`
4. Extract CEX: `cex-list ./prover_results`
5. Analyze: `cex-analyze ./prover_results rule_name`

You're ready! 🚀
