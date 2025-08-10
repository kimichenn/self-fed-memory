.PHONY: help venv install install-dev install-ui dev setup run api-dev ui-dev ingest-folder test test-cov test-manual test-manual-list lint format type-check check clean docker-build docker-up docker-down lock lock-upgrade

PYTHON ?= python
VENV_DIR ?= .venv
PORT_API ?= 8000
PORT_UI ?= 8501

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

venv: ## Create a local Python virtual environment in .venv (no-op if exists)
	@[ -d "$(VENV_DIR)" ] || $(PYTHON) -m venv $(VENV_DIR)
	@echo "Activate with: source $(VENV_DIR)/bin/activate"

install: ## Install the package (editable)
	@$(PYTHON) -m pip install -U pip
	@$(PYTHON) -m pip install -e .

install-dev: ## Install dev + test dependencies
	@$(PYTHON) -m pip install -U pip
	@$(PYTHON) -m pip install -e ".[dev,test]"

install-ui: ## Install UI dependencies (Streamlit / Gradio)
	@$(PYTHON) -m pip install -U pip
	@$(PYTHON) -m pip install -e ".[ui]"

dev: ## Install dev, test, and UI deps; set up git hooks; create .env if missing
	@$(PYTHON) -m pip install -U pip
	@($(PYTHON) -m pip show pip-tools >/dev/null 2>&1 || $(PYTHON) -m pip install pip-tools)
	@[ -f requirements-dev.txt ] || (echo "No requirements-dev.txt found; generating lock..." && $(MAKE) lock)
	@$(PYTHON) -m pip install -r requirements-dev.txt
	@$(PYTHON) -m pre_commit install
	@$(PYTHON) -m pre_commit install --hook-type pre-push
	@[ -f .env ] || (cp .env.example .env && echo ".env created from .env.example")
	@echo "Dev environment ready. Start the app with: make run"

# Backwards-compat alias
setup: install ## Install the package (editable)

api-dev: ## Start FastAPI dev server (uses current shell environment)
	@YAML_CEXT_DISABLED=1 $(PYTHON) -m uvicorn app.api.main:app --reload --host 0.0.0.0 --port $(PORT_API)

ui-dev: ## Start Streamlit UI (uses current shell environment)
	@YAML_CEXT_DISABLED=1 SELF_MEMORY_API=http://localhost:$(PORT_API) $(PYTHON) -m streamlit run frontend/app.py --server.port $(PORT_UI)

run: ## Start API ($(PORT_API)) and UI ($(PORT_UI)) together; Ctrl+C to stop
	@echo "Starting API on $(PORT_API) and UI on $(PORT_UI) using active environment..."
	@(YAML_CEXT_DISABLED=1 $(PYTHON) -m uvicorn app.api.main:app --reload --host 0.0.0.0 --port $(PORT_API) & echo $$! > .uvicorn.pid); \
	sleep 1; \
	(YAML_CEXT_DISABLED=1 SELF_MEMORY_API=http://localhost:$(PORT_API) $(PYTHON) -m streamlit run frontend/app.py --server.port $(PORT_UI) || true); \
	(test -f .uvicorn.pid && kill `cat .uvicorn.pid` 2>/dev/null || true); rm -f .uvicorn.pid

ingest-folder: ## Ingest a folder of Markdown documents (FOLDER=/path/to/notes)
	@$(PYTHON) scripts/ingest_folder.py $(FOLDER)

test: ## Run unit + integration tests (fast, hermetic)
	@$(PYTHON) -m pytest -m "unit or integration"

test-cov: ## Run tests with coverage report
	@$(PYTHON) -m pytest -m "unit or integration" --cov=app --cov-report=term-missing --cov-report=html:htmlcov

test-manual: ## Run manual tests (real APIs; requires API keys)
	@$(PYTHON) -m pytest -m manual -s --tb=short -x

test-manual-list: ## List available manual tests
	@$(PYTHON) -m pytest -q -m manual --collect-only | cat

lint: ## Lint with ruff
	@$(PYTHON) -m ruff check .

format: ## Format with ruff
	@$(PYTHON) -m ruff format .

type-check: ## Type-check with mypy
	@$(PYTHON) -m mypy .

check: ## Run format, lint, type-check, and tests
	@$(MAKE) format
	@$(MAKE) lint
	@$(MAKE) type-check
	@$(MAKE) test

clean: ## Remove build, cache, and coverage artifacts
	@rm -rf build/ dist/ *.egg-info/ htmlcov/ .coverage .pytest_cache/
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete

docker-build: ## Build Docker image
	@docker build -t self-fed-memory:latest .

docker-up: ## Run via Docker Compose
	@docker compose up --build

docker-down: ## Stop Docker Compose services
	@docker compose down

lock: ## Generate pinned requirements-dev.txt from pyproject extras (dev,test,ui)
	@$(PYTHON) -m pip install -U pip
	@($(PYTHON) -m pip show piptools >/dev/null 2>&1 || $(PYTHON) -m pip install pip-tools)
	@$(PYTHON) -m piptools compile --resolver=backtracking --extra dev --extra test --extra ui -o requirements-dev.txt $(CURDIR)/pyproject.toml

lock-upgrade: ## Recompute lock and upgrade pins
	@$(PYTHON) -m pip install -U pip
	@($(PYTHON) -m pip show piptools >/dev/null 2>&1 || $(PYTHON) -m pip install pip-tools)
	@$(PYTHON) -m piptools compile --resolver=backtracking --upgrade --extra dev --extra test --extra ui -o requirements-dev.txt $(CURDIR)/pyproject.toml
