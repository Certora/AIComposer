#!/usr/bin/env bash

# usage: source ./autoprover-setup.sh
# then run console-autoprove ...

set -eaux

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUTOSETUP_REPO="git@github.com:Certora/Autosetup.git"
PREAUDIT_REPO="git@github.com:Certora/PreAudit.git"
PREAUDIT_COMMIT="b941b4147d30fa460e2f9a593b1c8517e7e62eee"
AUTOSETUP_DIR="$SCRIPT_DIR/local/Autosetup"
PREAUDIT_DIR="$SCRIPT_DIR/local/PreAudit"
VENV_DIR="$SCRIPT_DIR/local/autoprover-venv"

if ! command -v git &>/dev/null; then
    echo "Error: git is not installed or not in PATH." >&2
    exit 1
fi

if ! command -v python3 &>/dev/null; then
    echo "Error: python3 is not installed or not in PATH." >&2
    exit 1
fi

if [[ -d "$AUTOSETUP_DIR" ]]; then
    echo "Autosetup already cloned, skipping."
else
    git clone --depth 1 "$AUTOSETUP_REPO" "$AUTOSETUP_DIR"
fi

if [[ -d "$PREAUDIT_DIR" ]]; then
    echo "PreAudit already cloned, skipping."
else
    git clone "$PREAUDIT_REPO" "$PREAUDIT_DIR"
fi

cd "$SCRIPT_DIR"

if [[ -d "$VENV_DIR" ]]; then
    echo "Venv already exists, skipping."
else
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

pip install -e "$AUTOSETUP_DIR"
cd "$PREAUDIT_DIR"
pip install -e .
git checkout "$PREAUDIT_COMMIT"
cd "$SCRIPT_DIR"
pip install solc-select

solc-select install latest
solc-select use latest

export AUTOSETUP_PATH="$AUTOSETUP_DIR"
export PREAUDIT_PATH="$PREAUDIT_DIR"

# we don't need to install ai-composer itself because autosetup and
# preaudit will both install it as a dependency. we must run
# autoprover from within the virtual env we've just created, though

set +eaux
echo "Done."
