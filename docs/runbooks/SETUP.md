# Repository setup

This runbook covers installing dependencies and verifying a local checkout of **moral-outrage-classifier**.

## Prerequisites

- **Python 3.12 or newer** (see `.python-version` and `requires-python` in `pyproject.toml`).
- **[uv](https://docs.astral.sh/uv/)** (recommended). The repo ships a `uv.lock` file for reproducible installs.

If you prefer not to use uv, you can use another environment manager and install from `pyproject.toml` (see [Alternative: pip](#alternative-pip)).

## Clone the repository

```bash
git clone <repository-url>
cd moral_outrage_classifier
```

Use your team’s actual clone URL in place of `<repository-url>`.

## Install dependencies (uv)

From the repository root:

```bash
uv sync
```

This creates a virtual environment (under `.venv` by default) and installs the locked dependency set.

## Run a quick sanity check

```bash
uv run python main.py
```

You should see: `Hello from moral-outrage-classifier!`

## Optional: activate the virtual environment

If you want an activated shell instead of prefixing commands with `uv run`:

```bash
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows (cmd/PowerShell)
python main.py
```

## Alternative: pip

If you do not use uv:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install "numpy>=2.4.3" "pandas>=3.0.1" "scikit-learn>=1.8.0"
python main.py
```

Exact versions may differ from the locked set in `uv.lock`; use `uv sync` when you need a reproducible environment.

## Troubleshooting

- **Wrong Python version**: Install Python 3.12+ and ensure `python --version` matches before installing.
- **`uv: command not found`**: Install uv from the [official docs](https://docs.astral.sh/uv/getting-started/installation/) or use the pip workflow above.
