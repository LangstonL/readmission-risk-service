.PHONY: install train test lint

install:
	python -m pip install -e ".[dev]"

train:
	python scripts/train_baseline.py

test:
	pytest -q

lint:
	ruff check .
