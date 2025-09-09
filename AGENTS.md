# Repository Guidelines

## Project Structure & Module Organization
- `src/graphrag_anthropic_llamaindex/`: core Python package (CLI in `main.py`, document/index ops, vector stores, search, utils).
- `tests/`: pytest suite (unit/integration), e.g., `tests/test_index_creation.py`.
- `data/`: input documents (you provide).
- `graphrag_output/`: generated indices and parquet outputs.
- `docker/`: Dockerfile, compose, `.env.example`.
- `gradio_app.py`: optional local UI.
- `Makefile`: common dev/ops commands.

## Build, Test, and Development Commands
- Setup (venv + install):
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -e .`
- Run CLI:
  - `python -m graphrag_anthropic_llamaindex.main add --config config.yaml`
  - `python -m graphrag_anthropic_llamaindex.main search "python graphs" --target-index both`
- Gradio app: `python gradio_app.py` (served on `http://localhost:7860`).
- Docker (preferred for UI/dev): `make up`, `make logs`, `make down`, `make shell`.
- Tests: `pytest -q` (runs under `tests/`).

## Coding Style & Naming Conventions
- Python ≥3.10. Use PEP 8, 4-space indentation, type hints where practical.
- Names: modules/functions/vars `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Keep functions short and single-responsibility; add docstrings for public APIs.
- Follow existing patterns in `vector_store_manager.py`, `document_processor.py`, and `search_processor.py`.

## Testing Guidelines
- Framework: pytest. Place new tests in `tests/` named `test_*.py` and functions `test_*`.
- Avoid external calls in tests. Mock LLMs using `llama_index.core.llms.mock.MockLLM` (see test setup) and use temp dirs for I/O.
- Validate parquet outputs and index directories rather than network responses.

## Commit & Pull Request Guidelines
- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:` (matches git history).
- PRs should include: clear description, rationale, screenshots/logs when UI/CLI behavior changes, and linked issues.
- Keep diffs focused; include tests for new behavior and updated docs when applicable.

## Security & Configuration Tips
- Do not commit secrets. Set `ANTHROPIC_API_KEY` via env or `docker/.env` (see `docker/.env.example`).
- Copy and edit `config.example.yaml` to `config.yaml`. For Bedrock, configure `bedrock` keys and AWS credentials.
- Large outputs live under `graphrag_output/`; keep them out of commits.
