.PHONY: help install install-dev test lint format clean train serve monitor docker-build docker-run

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $1, $2}'

install:  ## Install production dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run tests
	pytest tests/ -v --cov=src --cov-report=html

lint:  ## Run linting
	ruff check src tests
	black --check src tests

format:  ## Format code
	black src tests
	ruff --fix src tests

clean:  ## Clean cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

train:  ## Train the model
	python scripts/train_model.py --data-path data/raw/user_logs.jsonl

retrain:  ## Retrain the model
	python scripts/retrain_model.py --data-path data/raw/user_logs_new.jsonl

serve:  ## Start the API server
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

monitor:  ## Start monitoring
	python scripts/monitor_model.py

docker-build:  ## Build Docker image
	docker-compose build

docker-run:  ## Run with Docker Compose
	docker-compose up -d

docker-stop:  ## Stop Docker services
	docker-compose down

setup-data:  ## Setup sample data structure
	mkdir -p data/raw data/processed data/external
	mkdir -p models logs mlruns

mlflow-ui:  ## Start MLflow UI
	mlflow ui --host 0.0.0.0 --port 5000