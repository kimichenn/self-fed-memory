.PHONY: setup test test-cov clean lint format type-check install-dev help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install the package and dependencies
	@echo "Installing package in development mode..."
	@pip install -e .

install-dev: ## Install development dependencies
	@echo "Installing development dependencies..."
	@pip install -e ".[dev,test]"

test: ## Run tests
	@echo "Running tests..."
	@PYTHONPATH=. pytest

test-cov: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	@PYTHONPATH=. pytest --cov=app --cov-report=term-missing --cov-report=html

test-integration: ## Run integration tests only
	@echo "Running integration tests..."
	@PYTHONPATH=. pytest -m integration

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	@PYTHONPATH=. pytest -m "not integration"

lint: ## Run linting
	@echo "Running linter..."
	@ruff check .

format: ## Format code
	@echo "Formatting code..."
	@ruff format .

type-check: ## Run type checking
	@echo "Running type checker..."
	@mypy app/

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
	@echo "Development environment ready!"

# Ingestion shortcuts
ingest-folder: ## Ingest a folder of documents (usage: make ingest-folder FOLDER=/path/to/folder)
	@python scripts/ingest_folder.py $(FOLDER)

# API shortcuts  
api-dev: ## Start FastAPI development server
	@echo "Starting FastAPI development server..."
	@uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

ui-dev: ## Start Streamlit UI development server
	@echo "Starting Streamlit development server..."
	@streamlit run frontend/app.py 