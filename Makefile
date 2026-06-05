.DEFAULT_GOAL := help
.PHONY: help install install-dev run test cov lint format type check eval docker docker-up clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the app with all runtime extras (incl. embeddings/torch)
	pip install -e ".[app]"

install-dev:  ## Install dev/CI toolchain (no torch)
	pip install -e ".[dev]"

run:  ## Launch the Streamlit app
	streamlit run app.py

test:  ## Run the test suite
	pytest -q

cov:  ## Run tests with coverage
	pytest --cov=aria --cov-report=term-missing

lint:  ## Lint with ruff
	ruff check src tests

format:  ## Auto-format with ruff
	ruff format src tests
	ruff check --fix src tests

type:  ## Type-check with mypy
	mypy src

check: lint type cov  ## Run lint + type-check + coverage

eval:  ## Run the memory ablation harness and write a report
	python -m aria.eval.runner --out eval_reports

docker:  ## Build the Docker image
	docker build -t aria-memory-agent .

docker-up:  ## Run via docker compose
	docker compose up --build

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov build dist src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
