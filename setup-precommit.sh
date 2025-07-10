#!/bin/bash
# Pre-commit setup script for city2graph

echo "Setting up pre-commit for city2graph..."
echo "========================================"

# Install dependencies if needed
echo "Installing dependencies with uv..."
uv sync --group dev

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Run a focused test to verify setup
echo "Testing pre-commit on main source files..."
pre-commit run check-yaml --all-files
pre-commit run check-toml --all-files
pre-commit run end-of-file-fixer --files README.md pyproject.toml

echo ""
echo "✅ Pre-commit setup complete!"
echo ""
echo "The following hooks are now active:"
echo "  - File validation (YAML, TOML, JSON)"
echo "  - Code formatting with ruff"
echo "  - Type checking with mypy (for main source code)"
echo "  - Basic file hygiene (trailing whitespace, end-of-file)"
echo ""
echo "These will run automatically on git commit."
echo "To run manually: pre-commit run --all-files"
echo "To run on specific files: pre-commit run --files <file1> <file2>"
