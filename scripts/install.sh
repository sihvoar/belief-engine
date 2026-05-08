#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Bayes Tree — Linux / macOS installer
# Creates a virtual environment and installs all dependencies.
# ─────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
MIN_PYTHON="3.9"

# ── Colours ───────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${CYAN}ℹ ${NC}$*"; }
ok()    { echo -e "${GREEN}✓ ${NC}$*"; }
warn()  { echo -e "${YELLOW}⚠ ${NC}$*"; }
fail()  { echo -e "${RED}✗ ${NC}$*"; exit 1; }

echo
echo -e "${BOLD}🌳 Bayes Tree — Installer${NC}"
echo "─────────────────────────────────────────────"
echo

# ── Find Python ───────────────────────────────────────────────
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        major=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ] 2>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    fail "Python $MIN_PYTHON+ is required but not found.\n   Install it with: sudo apt install python3 python3-venv python3-pip"
fi
ok "Found Python $version ($PYTHON)"

# ── Check venv module ────────────────────────────────────────
if ! "$PYTHON" -m venv --help &>/dev/null; then
    fail "Python venv module not available.\n   Install it with: sudo apt install python3-venv"
fi

# ── Create virtual environment ────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment already exists at $VENV_DIR"
    read -rp "   Recreate it? [y/N] " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        info "Removing old virtual environment..."
        rm -rf "$VENV_DIR"
    else
        info "Keeping existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Virtual environment created"
fi

# ── Install dependencies ─────────────────────────────────────
info "Installing dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet
"$VENV_DIR/bin/pip" install -e "$SCRIPT_DIR[all]" --quiet
ok "Dependencies installed"

# ── Verify installation ──────────────────────────────────────
info "Verifying installation..."
"$VENV_DIR/bin/python" -c "
import yaml, matplotlib, numpy
from reportlab.lib.pagesizes import A4
from PyQt6.QtWidgets import QApplication
print('All packages OK')
" 2>/dev/null && ok "All packages verified" || warn "Some packages may not have loaded (display issues are OK)"

# ── Done ──────────────────────────────────────────────────────
echo
echo "─────────────────────────────────────────────"
echo -e "${GREEN}${BOLD}Installation complete!${NC}"
echo
echo "  To run the CLI:"
echo -e "    ${CYAN}$VENV_DIR/bin/python bayes-tree-eng.py examples/shroud.yaml${NC}"
echo
echo "  To run the GUI:"
echo -e "    ${CYAN}$VENV_DIR/bin/python bayes_tree_gui.py${NC}"
echo
echo "  Or activate the environment first:"
echo -e "    ${CYAN}source $VENV_DIR/bin/activate${NC}"
echo -e "    ${CYAN}python bayes_tree_gui.py${NC}"
echo
