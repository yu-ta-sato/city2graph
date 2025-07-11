# Pre-commit Setup for city2graph

This repository uses pre-commit hooks to maintain code quality and consistency.

## Setup

1. Install dependencies including pre-commit:
```bash
uv sync --group dev --extra cpu
```

2. Install the pre-commit hooks:
```bash
uv run pre-commit install
```

## What's Included

The pre-commit configuration includes:

- **File validation**: YAML, TOML, JSON syntax checking
- **Code formatting**: Automatic formatting with ruff
- **Code linting**: Style and error checking with ruff
- **Type checking**: Static type analysis with mypy (main source code only)
- **File hygiene**: Trailing whitespace, end-of-file fixes

## Usage

Pre-commit hooks run automatically on `git commit`. You can also run them manually:

```bash
# Run all hooks on all files
uv run pre-commit run --all-files

# Run specific hook on all files
uv run pre-commit run ruff-check --all-files

# Run hooks on specific files
uv run pre-commit run --files city2graph/utils.py city2graph/graph.py
```

## Configuration

The configuration is in `.pre-commit-config.yaml` and follows OSMnx patterns:
- Excludes notebooks and virtual environments from strict checking
- Uses the same tool versions and configurations as the main project
- Focuses on code quality for the main library code

## Troubleshooting

If pre-commit fails:
1. Read the error message carefully
2. Fix the issues manually or let auto-fixers handle them
3. Stage the changes and commit again

To bypass pre-commit (not recommended):
```bash
git commit --no-verify
```
