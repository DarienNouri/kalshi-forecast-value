.PHONY: setup check lint typecheck test validate-config build

setup:
	uv sync --locked --dev

check:
	uv lock --check
	uv run --locked ruff check .
	uv run --locked ty check src tests
	uv run --locked pytest
	uv run --locked kiq validate-config --config config/study.toml --json

lint:
	uv run --locked ruff check .

typecheck:
	uv run --locked ty check src tests

test:
	uv run --locked pytest

validate-config:
	uv run --locked kiq validate-config --config config/study.toml --json

build:
	uv build
