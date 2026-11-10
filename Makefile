UV_RUN ?= uv run

MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	$(UV_RUN) python -m src

debug:
	$(UV_RUN) python -m pdb -m src

clean:
	find . -path ./.venv -prune -o -type d \( -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache \) -prune -exec rm -rf {} +

lint:
	$(UV_RUN) flake8 .
	$(UV_RUN) mypy src $(MYPY_FLAGS)

lint-strict:
	$(UV_RUN) flake8 .
	$(UV_RUN) mypy src --strict
