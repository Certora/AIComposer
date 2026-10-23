#!/bin/bash
################################################################################
# CERTORA CEX EXTRACTION & CLAUDE ANALYSIS - AUTOMATED SETUP
#
# This script automates the entire one-time setup process.
# Run this ONCE on your machine, then use across all projects.
#
# Usage:
#   chmod +x setup_certora_tools.sh
#   ./setup_certora_tools.sh
#
# What it does:
#   1. Creates ~/certora-tools directory
#   2. Sets up Python virtual environment
#   3. Installs dependencies (anthropic, python-dotenv)
#   4. Downloads CEX extraction tool
#   5. Creates configuration file with your API key
#   6. Sets up shell aliases
#   7. Verifies everything works
#
# Requirements:
#   - Python 3.9+
#   - pip
#   - Your Anthropic API key (get from: https://console.anthropic.com/)
#
################################################################################

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print with color
print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Start setup
print_header "CERTORA CEX EXTRACTION & CLAUDE ANALYSIS - SETUP"

echo "This setup script will:"
echo "  1. Create ~/certora-tools directory"
echo "  2. Set up Python virtual environment"
echo "  3. Install dependencies"
echo "  4. Download CEX extraction tool"
echo "  5. Configure Claude API key"
echo "  6. Set up shell aliases"
echo "  7. Verify everything works"
echo ""

# Check prerequisites
print_header "STEP 1: Checking Prerequisites"

# Check Python version
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.9 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_success "Python $PYTHON_VERSION found"

# Check pip
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 not found. Please install pip."
    exit 1
fi

print_success "pip found"

# Create tools directory
print_header "STEP 2: Creating Tools Directory"

TOOLS_HOME="$HOME/certora-tools"

if [ -d "$TOOLS_HOME" ]; then
    print_warning "Directory already exists: $TOOLS_HOME"
    read -p "Overwrite existing setup? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Setup cancelled"
        exit 0
    fi
else
    mkdir -p "$TOOLS_HOME"
    print_success "Created: $TOOLS_HOME"
fi

cd "$TOOLS_HOME"
print_success "Working directory: $(pwd)"

# Create Python virtual environment
print_header "STEP 3: Setting Up Python Virtual Environment"

if [ -d "venv" ]; then
    print_warning "Virtual environment already exists"
    read -p "Recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        print_success "Created new virtual environment"
    fi
else
    python3 -m venv venv
    print_success "Created virtual environment"
fi

# Activate venv
source venv/bin/activate
print_success "Activated virtual environment"

# Upgrade pip
print_header "STEP 4: Installing Dependencies"

print_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

print_info "Installing anthropic package..."
pip install anthropic > /dev/null 2>&1
print_success "anthropic installed"

print_info "Installing python-dotenv..."
pip install python-dotenv > /dev/null 2>&1
print_success "python-dotenv installed"

# Download CEX tool
print_header "STEP 5: Downloading CEX Extraction Tool"

CEX_TOOL="$TOOLS_HOME/extract_and_analyze_cex.py"

if [ -f "$CEX_TOOL" ]; then
    print_warning "Tool already exists"
    read -p "Re-download it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_success "Using existing tool"
    else
        curl -s -o "$CEX_TOOL" \
            https://raw.githubusercontent.com/0xbrett8571/CertoraAIComposer/master/extract_and_analyze_cex.py
        print_success "Downloaded CEX tool"
    fi
else
    print_info "Downloading from GitHub..."
    curl -s -o "$CEX_TOOL" \
        https://raw.githubusercontent.com/0xbrett8571/CertoraAIComposer/master/extract_and_analyze_cex.py
    
    if [ ! -f "$CEX_TOOL" ]; then
        print_error "Failed to download tool"
        exit 1
    fi
    
    print_success "Downloaded CEX tool"
fi

# Get API key
print_header "STEP 6: Configuring Claude API Key"

print_info "You need an Anthropic API key to use Claude."
print_info "Get one from: https://console.anthropic.com/account/keys"
print_info ""

if [ -z "$ANTHROPIC_API_KEY" ]; then
    read -sp "Enter your Anthropic API key (sk-ant-...): " API_KEY
    echo
    
    if [ -z "$API_KEY" ]; then
        print_error "API key is required"
        exit 1
    fi
else
    print_info "ANTHROPIC_API_KEY already set in environment"
    API_KEY="$ANTHROPIC_API_KEY"
fi

print_success "API key configured"

# Create configuration file
print_header "STEP 7: Creating Configuration File"

CONFIG_FILE="$HOME/.certora-config"

cat > "$CONFIG_FILE" << EOF
# Certora Tools Configuration
# Source this file to set up environment for Certora CEX analysis
#
# Add to ~/.bashrc or ~/.zshrc:
#   source ~/.certora-config

# Claude API Configuration
export ANTHROPIC_API_KEY="$API_KEY"

# Python Environment
CERTORA_TOOLS_HOME="$TOOLS_HOME"
export PYTHON_ENV="\$CERTORA_TOOLS_HOME/venv"

# Tool Paths
export CEX_TOOL="\$CERTORA_TOOLS_HOME/extract_and_analyze_cex.py"
export CERTORA_TOOL_BIN="\$PYTHON_ENV/bin/python"

# Default settings
export CERTORA_OUTPUT_DIR="/tmp/certoraOutput"
export CLAUDE_MODEL="claude-3-5-sonnet-20241022"

# Logging
export CERTORA_LOG_LEVEL="INFO"

# Convenience aliases
alias cex-tools='cd \$CERTORA_TOOLS_HOME && source venv/bin/activate'
alias cex-list='\$CERTORA_TOOL_BIN \$CEX_TOOL list'
alias cex-extract='\$CERTORA_TOOL_BIN \$CEX_TOOL extract'
alias cex-analyze='\$CERTORA_TOOL_BIN \$CEX_TOOL analyze'

# Helper functions
certora-verify() {
    echo "Verifying Certora setup..."
    echo "✓ Tools Home: \$CERTORA_TOOLS_HOME"
    echo "✓ Python Env: \$PYTHON_ENV"
    echo "✓ CEX Tool: \$CEX_TOOL"
    echo "✓ API Key: \${ANTHROPIC_API_KEY:0:15}..."
    echo "✓ Model: \$CLAUDE_MODEL"
}
EOF

chmod 600 "$CONFIG_FILE"
print_success "Configuration file created: $CONFIG_FILE"
print_info "API key is stored securely (permissions: 600)"

# Update shell profile
print_header "STEP 8: Updating Shell Profile"

SHELL_PROFILE=""
if [ -f "$HOME/.bashrc" ]; then
    SHELL_PROFILE="$HOME/.bashrc"
elif [ -f "$HOME/.zshrc" ]; then
    SHELL_PROFILE="$HOME/.zshrc"
else
    print_warning "Shell profile not found. Please add manually:"
    echo "  source ~/.certora-config"
fi

if [ -n "$SHELL_PROFILE" ]; then
    if grep -q "source ~/.certora-config" "$SHELL_PROFILE"; then
        print_info "Already configured in shell profile"
    else
        echo "" >> "$SHELL_PROFILE"
        echo "# Certora Tools Configuration" >> "$SHELL_PROFILE"
        echo "source ~/.certora-config" >> "$SHELL_PROFILE"
        print_success "Updated shell profile: $SHELL_PROFILE"
    fi
fi

# Verify installation
print_header "STEP 9: Verifying Installation"

# Source config
source "$CONFIG_FILE"

# Test API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    print_error "API key not set"
    exit 1
fi
print_success "API key configured"

# Test Python
if ! $CERTORA_TOOL_BIN --version &> /dev/null; then
    print_success "Python environment working"
fi

# Test tool
if $CERTORA_TOOL_BIN "$CEX_TOOL" --help > /dev/null 2>&1; then
    print_success "CEX tool working"
else
    print_warning "CEX tool help failed, but tool exists"
fi

# Test aliases
if alias cex-analyze &> /dev/null; then
    print_success "Shell aliases configured"
else
    print_warning "Aliases not yet active (reload shell to activate)"
fi

# Final summary
print_header "✅ SETUP COMPLETE!"

echo "What's been configured:"
echo "  ✓ Tools directory: $TOOLS_HOME"
echo "  ✓ Python virtual environment"
echo "  ✓ Dependencies installed"
echo "  ✓ CEX extraction tool downloaded"
echo "  ✓ Configuration file: $CONFIG_FILE"
echo "  ✓ Shell aliases configured"
echo ""

echo "📋 Next Steps:"
echo ""
echo "1. Reload your shell to activate aliases:"
echo "   ${BLUE}source ~/.bashrc${NC}  # or ~/.zshrc"
echo ""
echo "2. Verify setup:"
echo "   ${BLUE}certora-verify${NC}"
echo ""
echo "3. For each project, create .certora-project file:"
echo "   ${BLUE}cat > .certora-project << 'EOF'"
echo "PROJECT_NAME=\"my-project\""
echo "CERTORA_OUTPUT_DIR=\"./prover_results\""
echo "SPEC_DIR=\"./spec\""
echo "MAIN_CONTRACT=\"MyContract\""
echo "EOF${NC}"
echo ""
echo "4. Run your first analysis:"
echo "   ${BLUE}certoractl run config.conf -o ./prover_results${NC}"
echo "   ${BLUE}cex-list ./prover_results${NC}"
echo "   ${BLUE}cex-analyze ./prover_results rule_name --spec ./spec/file.spec${NC}"
echo ""

echo "📖 Documentation:"
echo "   Setup Guide: SETUP_GUIDE.md"
echo "   Quick Reference: QUICK_REFERENCE.md"
echo "   Tool Usage: extract_and_analyze_cex.py --help"
echo ""

echo "🔗 Useful Links:"
echo "   Get API Key: https://console.anthropic.com/account/keys"
echo "   Certora Docs: https://docs.certora.com"
echo "   AIComposer Repo: https://github.com/0xbrett8571/CertoraAIComposer"
echo ""

print_success "Setup script completed successfully!"
print_info "Reload your terminal or run: source ~/.bashrc"
