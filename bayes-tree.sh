#!/usr/bin/env bash
# Bayes Tree launcher — auto-installs and runs the GUI or CLI
# Usage:
#   ./bayes-tree.sh                          # launch GUI
#   ./bayes-tree.sh examples/shroud.yaml     # GUI with file
#   ./bayes-tree.sh --cli examples/shroud.yaml  # CLI mode
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv"

# Auto-install if needed
if [ ! -d "$VENV" ]; then
    echo "First run — installing dependencies..."
    "$DIR/install.sh"
    echo
fi

PY="$VENV/bin/python"

if [ "$1" = "--cli" ]; then
    shift
    exec "$PY" "$DIR/bayes-tree-eng.py" "$@"
else
    exec "$PY" "$DIR/bayes_tree_gui.py" "$@"
fi
