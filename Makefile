.PHONY: setup test test-cov clean lint format type-check install-dev help dev run venv docker-build docker-up docker-down test-manual-list

# Tool-agnostic defaults (works with system Python, venv, or conda)
PYTHON ?= python3
VENV_DIR ?= .venv

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

venv: ## Create a local Python virtual environment in .venv (if missing)
	@echo "Ensuring virtual environment exists at $(VENV_DIR) ..."
	@([ -d "$(VENV_DIR)" ] || ($(PYTHON) -m venv $(VENV_DIR)))
	@echo "Virtual environment ready. Activate with: source $(VENV_DIR)/bin/activate"

dev: venv ## One-step local dev setup (venv + dev/test/ui deps + hooks)
	@echo "Installing development, test, and UI dependencies into $(VENV_DIR) ..."
	@. $(VENV_DIR)/bin/activate; pip install -U pip
	@. $(VENV_DIR)/bin/activate; pip install -e ".[dev,test,ui]"
	@echo "Installing git hooks (pre-commit)..."
	@. $(VENV_DIR)/bin/activate; pre-commit install
	@. $(VENV_DIR)/bin/activate; pre-commit install --hook-type pre-push
	@([ -f ".env" ] || (cp .env.example .env && echo ".env created from .env.example")) || true
	@echo "Dev environment ready. To start the app, run: make run"

run: ## Start API and UI together (Ctrl+C to stop)
	@echo "Starting FastAPI (8000) and Streamlit UI (8501)..."
	@(. $(VENV_DIR)/bin/activate 2>/dev/null || true; \
	 uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000 & echo $$! > .uvicorn.pid); \
	 sleep 1; \
	 (SELF_MEMORY_API=http://localhost:8000 streamlit run frontend/app.py || true); \
	 (test -f .uvicorn.pid && kill `cat .uvicorn.pid` 2>/dev/null || true); rm -f .uvicorn.pid

setup: ## Install the package and dependencies
	@echo "Installing package in development mode..."
	@pip install -e .

install-dev: ## Install development dependencies
	@echo "Installing development dependencies..."
	@pip install -e ".[dev,test]"

test: ## Run all automated (unit and integration) tests
	@echo "Running automated unit and integration tests..."
	@(. $(VENV_DIR)/bin/activate 2>/dev/null || true; PYTHONPATH=. pytest -m "unit or integration")

test-cov: ## Run automated tests with coverage report
	@echo "Running automated tests with coverage..."
	@(. $(VENV_DIR)/bin/activate 2>/dev/null || true; PYTHONPATH=. pytest -m "unit or integration" --cov=app --cov-report=term-missing --cov-report=html)

test-manual: ## Run manual verification tests (requires API keys)
	@echo "Running manual verification tests..."
	@(. $(VENV_DIR)/bin/activate 2>/dev/null || true; PYTHONPATH=. pytest -m "manual" -s --tb=short -x)

lint: ## Run linting
	@echo "Running linter..."
	@(. $(VENV_DIR)/bin/activate 2>/dev/null || true; ruff check .)

format: ## Format code
	@echo "Formatting code..."
	@(. $(VENV_DIR)/bin/activate 2>/dev/null || true; ruff format .)

type-check: ## Run type checking
	@echo "Running type checker..."
	@(. $(VENV_DIR)/bin/activate 2>/dev/null || true; mypy app/)

clean: ## Clean up build artifacts and cache
	@echo "Cleaning up..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf htmlcov/
	@rm -rf .coverage
	@rm -rf .pytest_cache/
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete

check: lint type-check test ## Run all checks (lint, type-check, test)

install-ui: ## Install UI dependencies for Streamlit/Gradio
	@echo "Installing UI dependencies..."
	@pip install -e ".[ui]"

dev-setup: install-dev ## Complete development setup
	@echo "Setting up development environment..."
	@pre-commit install
	@pre-commit install --hook-type pre-push
	@echo "Development environment ready!"

# Ingestion shortcuts
ingest-folder: ## Ingest a folder of documents (usage: make ingest-folder FOLDER=/path/to/folder)
	@(. $(VENV_DIR)/bin/activate 2>/dev/null || true; python scripts/ingest_folder.py $(FOLDER))

# API shortcuts
api-dev: ## Start FastAPI development server
	@echo "Starting FastAPI development server..."
	@(. $(VENV_DIR)/bin/activate 2>/dev/null || true; uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000)

ui-dev: ## Start Streamlit UI development server
	@echo "Starting Streamlit development server..."
	@(. $(VENV_DIR)/bin/activate 2>/dev/null || true; streamlit run frontend/app.py)

test-manual-list: ## List available manual tests
	@echo "Listing manual tests (pytest collection):"
	@PYTHONPATH=. pytest -q -m manual --collect-only | cat

# Docker helpers (optional)
docker-build: ## Build Docker image for API/UI
	@docker build -t self-fed-memory:latest .

docker-up: ## Run API (8000) and UI (8501) via Docker Compose
	@docker compose up --build

docker-down: ## Stop Docker Compose services
	@docker compose down
