# **CERTORA CEX ANALYSIS - QUICK REFERENCE CARD**

**Print this page and keep it on your desk!**

---

## **🚀 ONE-TIME SETUP (Do This Once)**

```bash
# 1. Create tools directory
mkdir -p ~/certora-tools && cd ~/certora-tools

# 2. Python environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install anthropic python-dotenv

# 4. Download tool
curl -o extract_and_analyze_cex.py \
  https://raw.githubusercontent.com/0xbrett8571/CertoraAIComposer/master/extract_and_analyze_cex.py

# 5. Create ~/.certora-config with:
cat > ~/.certora-config << 'EOF'
export ANTHROPIC_API_KEY="sk-ant-YOUR-KEY-HERE"
export CERTORA_TOOLS_HOME="$HOME/certora-tools"
export PYTHON_ENV="$CERTORA_TOOLS_HOME/venv"
export CEX_TOOL="$CERTORA_TOOLS_HOME/extract_and_analyze_cex.py"
export CERTORA_TOOL_BIN="$PYTHON_ENV/bin/python"
export CLAUDE_MODEL="claude-3-5-sonnet-20241022"

alias cex-list='$CERTORA_TOOL_BIN $CEX_TOOL list'
alias cex-extract='$CERTORA_TOOL_BIN $CEX_TOOL extract'
alias cex-analyze='$CERTORA_TOOL_BIN $CEX_TOOL analyze'
EOF

# 6. Add to ~/.bashrc or ~/.zshrc:
echo 'source ~/.certora-config' >> ~/.bashrc
source ~/.bashrc

# 7. Update with your API key
sed -i 's/sk-ant-YOUR-KEY-HERE/sk-ant-ACTUAL-KEY/' ~/.certora-config

# 8. Verify
echo $ANTHROPIC_API_KEY  # Should show your key
cex-list --help         # Should work
```

---

## **📋 PER-PROJECT SETUP (2 Minutes)**

In each new project, create `.certora-project`:

```bash
cat > .certora-project << 'EOF'
PROJECT_NAME="my-project"
CERTORA_OUTPUT_DIR="./prover_results"
SPEC_DIR="./spec"
MAIN_CONTRACT="MyContract"
EOF
```

**That's it! You're ready to use.**

---

## **⚡ Daily Commands**

### **1. Run Prover**
```bash
certoraRun config.conf --output_dir ./prover_results
# Wait for completion (5 min - 2 hours)
```

### **2. List All Rules**
```bash
cex-list ./prover_results

# Output shows: Rule Name | Status | CEX Available
# VIOLATED = has counterexample, VERIFIED = passed
```

### **3. Extract a CEX**
```bash
cex-extract ./prover_results rule_name --output cex.txt
cat cex.txt
```

### **4. Analyze with Claude (The Magic!)**
```bash
# Basic analysis
cex-analyze ./prover_results rule_name

# With spec context (recommended)
cex-analyze ./prover_results rule_name --spec ./spec/file.spec

# Save to file
cex-analyze ./prover_results rule_name \
  --spec ./spec/file.spec \
  --output analysis.md
```

### **5. Read the Analysis**
```bash
cat analysis.md
# Shows: Root cause, potential fixes, confidence level
```

---

## **🔄 Typical Workflow**

```
┌─ MORNING ─────────────────────────┐
│ 1. certoraRun config.conf         │ (run prover)
│    # Get coffee ☕ (wait)         │
└───────────────────────────────────┘
            ↓
┌─ MID-MORNING ─────────────────────┐
│ 2. cex-list ./prover_results      │ (see failures)
│    # Find VIOLATED rules           │
└───────────────────────────────────┘
            ↓
┌─ LATE MORNING ────────────────────┐
│ 3. cex-analyze ./prover_results \  │ (get analysis)
│      rule_name \                   │
│      --spec ./spec/file.spec \     │
│      --output analysis.md          │
│                                    │
│ 4. cat analysis.md                │ (read fixes)
└───────────────────────────────────┘
            ↓
┌─ NOON ────────────────────────────┐
│ 5. vim contracts/MyContract.sol   │ (apply fix)
│    # Edit based on Claude's suggestion
└───────────────────────────────────┘
            ↓
┌─ AFTERNOON ───────────────────────┐
│ 6. certoraRun config.conf         │ (re-verify)
│ 7. cex-list ./prover_results      │ (check if fixed)
│    # If still VIOLATED, go to step 3
└───────────────────────────────────┘
```

---

## **📊 Batch Analysis (Analyze All Failures at Once)**

```bash
#!/bin/bash
# Save as scripts/analyze_all.sh, then: chmod +x scripts/analyze_all.sh

source ~/.certora-config

python $CERTORA_TOOL_BIN $CEX_TOOL list ./prover_results | \
  grep VIOLATED | awk '{print $1}' | while read rule; do
    $CERTORA_TOOL_BIN $CEX_TOOL analyze ./prover_results "$rule" \
      --spec ./spec/*.spec \
      --output "analysis_${rule}.md"
    echo "✓ Analyzed: $rule"
done

echo "✅ All analyses saved!"
ls -lh analysis_*.md
```

**Run it:**
```bash
chmod +x scripts/analyze_all.sh
./scripts/analyze_all.sh
```

---

## **🐛 Troubleshooting**

| Problem | Solution |
|---------|----------|
| "ANTHROPIC_API_KEY not set" | `source ~/.certora-config` then `echo $ANTHROPIC_API_KEY` |
| "Rule not found" | `cex-list ./prover_results` to see exact names |
| "No CEX found" | Only VIOLATED rules have CEX. Check status in list |
| "Python not found" | `source ~/certora-tools/venv/bin/activate` |
| "Permission denied" | `chmod +x scripts/*.sh` |
| "Module not found" | Reinstall: `pip install anthropic` |

---

## **💡 Pro Tips**

### **Tip 1: Save Analyses in Git**
```bash
git add analysis_*.md
git commit -m "CEX analysis for iteration 1"
# Track progress over time
```

### **Tip 2: Compare Before/After**
```bash
# Before fix
cex-analyze ./results_v1 rule_name --output before.md

# After fix
cex-analyze ./results_v2 rule_name --output after.md

# Compare
diff before.md after.md
```

### **Tip 3: Watch Specific Rule**
```bash
# Keep checking one rule until it passes
while true; do
    STATUS=$(cex-list ./prover_results | grep transfer_succeeds | awk '{print $2}')
    echo "$(date): transfer_succeeds = $STATUS"
    sleep 60
done
```

### **Tip 4: Generate HTML Report**
```bash
# Convert markdown to HTML
pandoc analysis_*.md -o report.html

# Open in browser
open report.html
```

---

## **📁 File Locations Quick Reference**

| Item | Location |
|------|----------|
| Global Config | `~/.certora-config` |
| Tools Home | `~/certora-tools/` |
| Python Env | `~/certora-tools/venv/` |
| Tool Script | `~/certora-tools/extract_and_analyze_cex.py` |
| Project Config | `./. certora-project` |
| Prover Results | `./prover_results/` |
| Analyses | `./analysis_*.md` |
| Certora Config | `./config.conf` |

---

## **🎯 Key Variables**

```bash
# Show all settings
source ~/.certora-config
echo "Tools: $CERTORA_TOOLS_HOME"
echo "Tool: $CEX_TOOL"
echo "Python: $CERTORA_TOOL_BIN"
echo "Model: $CLAUDE_MODEL"
echo "API Key: ${ANTHROPIC_API_KEY:0:10}..." # First 10 chars
```

---

## **🚀 Super Quick Start (For Existing Setup)**

```bash
# Everything in 3 commands:

# 1. Run prover
certoraRun config.conf --output_dir ./prover_results

# 2. See what failed
cex-list ./prover_results | grep VIOLATED

# 3. Analyze first failure
cex-analyze ./prover_results [rule_name] \
  --spec ./spec/main.spec \
  --output analysis.md
```

---

## **💬 Common Patterns**

### **Pattern 1: Fix All Failures Systematically**
```bash
for rule in $(cex-list ./prover_results | grep VIOLATED | awk '{print $1}'); do
    cex-analyze ./prover_results "$rule" --spec ./spec/*.spec --output analysis_$rule.md
    echo "Review analysis_$rule.md and fix code"
    read -p "Press ENTER after fixing..."
done
```

### **Pattern 2: Monitor Until All Pass**
```bash
iteration=0
while [ $(cex-list ./prover_results_$iteration | grep VIOLATED | wc -l) -gt 0 ]; do
    iteration=$((iteration + 1))
    echo "Iteration $iteration..."
    certoraRun config.conf --output_dir ./prover_results_$iteration
done
echo "✅ All verified in $iteration iterations!"
```

### **Pattern 3: Compare Rules Across Versions**
```bash
# Version 1
cex-list ./v1_results | grep VIOLATED > v1_failures.txt

# Version 2  
cex-list ./v2_results | grep VIOLATED > v2_failures.txt

# Compare
diff v1_failures.txt v2_failures.txt
```

---

## **⚙️ Environment Variables (Advanced)**

```bash
# Edit ~/.certora-config to customize:

ANTHROPIC_API_KEY        # Your Claude API key
CLAUDE_MODEL             # Which Claude model to use
CERTORA_OUTPUT_DIR       # Default output location
LOOP_ITER                # Prover loop iterations
HASHING_BOUND            # Prover hashing bound
CERTORA_TOOL_BIN         # Python binary path
CEX_TOOL                 # Tool script location
```

---

## **📞 Need Help?**

1. Check troubleshooting section above
2. Run: `cex-analyze --help` for tool options
3. Review: `cat ~/.certora-config` for settings
4. Verify: `echo $ANTHROPIC_API_KEY` to check API key

---

## **✅ Verification Checklist**

Before you start:
- [ ] `echo $ANTHROPIC_API_KEY` shows your key
- [ ] `cex-list --help` works without errors
- [ ] `source ~/.certora-config` completes silently
- [ ] `ls ~/certora-tools/extract_and_analyze_cex.py` exists
- [ ] `.certora-project` file in your project
- [ ] `config.conf` in your project

All checked? **You're ready to roll!** 🎉

---

**Last Updated:** 2025-01-15  
**Tested On:** macOS, Linux, WSL  
**Requirements:** Python 3.9+, pip, Anthropic API key

**Keep this card handy!**
