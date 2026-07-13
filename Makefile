PYTHON ?= python3

.PHONY: install test audit

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

audit:
	$(PYTHON) -m evaltrust.cli audit examples/clean_win.json
