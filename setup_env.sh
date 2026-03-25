#!/bin/bash
# Setup script to replicate this environment from scratch
# Installs: Node.js, npm, Claude Code, uv, and Python dependencies

set -e

echo "=== Environment Setup ==="

# ── Node.js & npm ────────────────────────────────────────────────────────────
echo ""
echo "[1/4] Installing Node.js v20 and npm..."

if command -v node &>/dev/null; then
    echo "  Node.js already installed: $(node --version)"
else
    if command -v apt-get &>/dev/null; then
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt-get install -y nodejs
    elif command -v brew &>/dev/null; then
        brew install node@20
    else
        echo "  ERROR: Unsupported package manager. Install Node.js v20 manually."
        exit 1
    fi
    echo "  Node.js installed: $(node --version)"
fi

# ── Claude Code ──────────────────────────────────────────────────────────────
echo ""
echo "[2/4] Installing Claude Code..."

if command -v claude &>/dev/null; then
    echo "  Claude Code already installed: $(claude --version)"
else
    npm install -g @anthropic-ai/claude-code
    echo "  Claude Code installed: $(claude --version)"
fi

# ── uv ──────────────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Installing uv..."

if command -v uv &>/dev/null; then
    echo "  uv already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for the rest of this script
    export PATH="$HOME/.local/bin:$PATH"
    echo "  uv installed: $(uv --version)"
fi

# ── Python dependencies via uv ───────────────────────────────────────────────
echo ""
echo "[4/4] Installing Python dependencies from requirements.txt..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "  ERROR: requirements.txt not found at $SCRIPT_DIR/requirements.txt"
    exit 1
fi

uv pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Versions installed:"
echo "  Node.js : $(node --version)"
echo "  npm     : $(npm --version)"
echo "  Claude  : $(claude --version 2>/dev/null || echo 'restart shell to use')"
echo "  uv      : $(uv --version)"
echo "  Python  : $(python3 --version)"
echo ""
echo "NOTE: If uv was freshly installed, run: source \$HOME/.local/bin/env"
echo "      or open a new shell for PATH changes to take effect."
