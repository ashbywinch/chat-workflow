# Makefile for chat-workflow — single dev entry point.
#
# DO NOT CHANGE:
# - SHELL + .SHELLFLAGS (the shell pin): make's default /bin/sh is dash on
#   Debian/Ubuntu — including the CI runners — and dash has no `pipefail`.
#   Without the pin, a bash-ism passes locally and dies in CI with
#   "Illegal option -o pipefail".
# - `deps` never depends on `install-hooks`: a hook that calls `make check`
#   would re-copy/refuse the very hook file (the pre-push deadlock).
# - `check` = lint-check + typecheck — the single gate CI and pre-push both run.
# - `clean` never touches user data (only .venv + generated reports).

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: help setup deps uv-sync install-hooks run test test-unit test-verbose test-all check lint lint-check lint-github typecheck typecheck-update-baseline format coverage docs-check evals evals-verbose evals-smoke evals-incremental clean

# Variables
PYTHON := .venv/bin/python
UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
OUT := .venv/bin/chat-workflow
RUFF := .venv/bin/ruff
UNITTEST := $(PYTHON) -m unittest
COVERAGE := .venv/bin/coverage

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help:
	@echo "Available commands:"
	@echo "  ${GREEN}make setup${NC}              Create venv, install deps + hooks, ensure .env exists"
	@echo "  ${GREEN}make check${NC}              Lint + typecheck — the gate CI and the pre-push hook run"
	@echo "  ${GREEN}make test${NC}               Run unit tests (no API key required)"
	@echo "  ${GREEN}make test-verbose${NC}       Run unit tests with verbose output"
	@echo "  ${GREEN}make test-all${NC}           Run unit tests + evals"
	@echo "  ${GREEN}make evals${NC}              Run evaluation tests with real API (requires gateway token)"
	@echo "  ${GREEN}make evals-smoke${NC}        Quick eval subset (test_real_api + debug_streaming, ~80s)"
	@echo "  ${GREEN}make evals-incremental${NC}  Change-aware eval subset (auto-detects affected evals)"
	@echo "  ${GREEN}make coverage${NC}           Run tests with coverage report (coverage.xml for CI)"
	@echo "  ${GREEN}make docs-check${NC}         Every relative markdown link resolves"
	@echo "  ${GREEN}make lint${NC}               Check code quality with ruff"
	@echo "  ${GREEN}make format${NC}             Auto-fix linting issues"
	@echo "  ${GREEN}make typecheck${NC}          Pyrefly, gated on the committed baseline"
	@echo "  ${GREEN}make clean${NC}              Clean up generated files (removes .venv)"

setup: deps install-hooks
	@[ -f .env ] || cp .env.example .env
	@echo "${GREEN}✓ Setup complete${NC}"

# What CHECK targets actually need — no hook machinery. A check must never
# depend on install-hooks (see the deadlock note at the top).
deps: uv-sync

uv-sync:
	@$(UV) --version >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	@$(UV) sync --all-extras

install-hooks:
	@HOOKS_DIR=$$(git rev-parse --git-path hooks); \
	mkdir -p "$$HOOKS_DIR"; \
	for hook in pre-commit pre-push; do \
		if [ -f "$$HOOKS_DIR/$$hook" ] && ! cmp -s scripts/$$hook "$$HOOKS_DIR/$$hook"; then \
			echo "${YELLOW}Overwriting an existing $$hook hook (scripts/ is the single source of truth — re-run after pulling updates)${NC}"; \
		fi; \
		cp scripts/$$hook "$$HOOKS_DIR/$$hook"; \
		chmod +x "$$HOOKS_DIR/$$hook"; \
	done
	@echo "${GREEN}✓ Hooks installed (pre-commit: lint + gitleaks; pre-push: make check)${NC}"

run: setup
	@$(OUT) evaluation-criteria generate-reviewed-criteria --context "A birthday present"

# The exact gate a push must pass — run identically by CI and the pre-push
# hook (single source of truth; no test run — too slow for a hook, and CI's
# `make test` already includes these).
check: deps lint-check typecheck

test: deps lint-check typecheck
	@$(UNITTEST) discover tests/unit/

test-unit: test

test-verbose: deps lint-check typecheck
	@$(UNITTEST) discover tests/unit/ -v

test-all: test evals

coverage: deps
	@$(COVERAGE) run -m unittest discover tests/unit/
	@$(COVERAGE) report -m
	@$(COVERAGE) xml
	@$(COVERAGE) html
	@echo "${GREEN}Coverage report generated: htmlcov/index.html${NC}"

docs-check: deps  # every relative markdown link resolves (house standard)
	@$(UNITTEST) tests.unit.test_docs_links

lint: deps lint-check

lint-check: deps  # Shared with the pre-commit hook — single source of truth for the lint scope
	@$(RUFF) check chat_workflow/ workflows/ tests/ chat_workflow_cli/

lint-github: deps  # CI only: findings surface as PR annotations
	@$(RUFF) check chat_workflow/ workflows/ tests/ chat_workflow_cli/ --output-format=github

# pyrefly + BOTH-direction baseline lock (new errors AND stale entries fail)
typecheck: deps
	@$(PYTHON) scripts/pyrefly-lock.py check --pyrefly-config pyrefly.toml

# After a deliberate diagnostic change, commit the refresh
typecheck-update-baseline: deps
	@$(PYTHON) scripts/pyrefly-lock.py update-baseline --pyrefly-config pyrefly.toml

# Auto-fix linting issues
format: setup
	@$(RUFF) check --fix chat_workflow/ workflows/ tests/ chat_workflow_cli/
	@$(RUFF) format chat_workflow/ workflows/ tests/ chat_workflow_cli/

# Evals call the real LLM through Cloudflare AI Gateway. `load_dotenv()` in
# the library reads .env (created by `make setup`); real env vars win.
evals: deps lint-check typecheck
	@rm -f test-results/eval-report.txt
	@$(PYTHON) scripts/run_with_timeout.py --timeout 1800 -- $(UNITTEST) discover tests/evals/

evals-verbose: deps lint-check typecheck
	@$(PYTHON) scripts/run_with_timeout.py --timeout 1800 -- $(UNITTEST) discover tests/evals/ -v

evals-smoke: deps lint-check typecheck
	@$(PYTHON) scripts/run_with_timeout.py --timeout 600 -- $(UNITTEST) tests.evals.single.test_real_api tests.evals.composite.test_debug_streaming_api -v

evals-incremental: deps lint-check typecheck
	@echo "${YELLOW}Updating dependency graph...${NC}"
	@$(PYTHON) -m code_review_graph update 2>&1 | grep -v "^$$" || true
	@FILES=$$($(PYTHON) scripts/affected_evals.py --git-base origin/main); \
	COUNT=$$(echo $$FILES | wc -w); \
	TOTAL=$$($(PYTHON) -c "import glob; print(len(glob.glob('tests/evals/**/test_*.py', recursive=True)))"); \
	if [ -z "$$FILES" ]; then \
		echo "${GREEN}No evals affected by current changes.${NC}"; \
	else \
		echo "${YELLOW}Running $$COUNT of $$TOTAL evals...${NC}"; \
		$(PYTHON) scripts/run_with_timeout.py --timeout 1800 -- $(UNITTEST) $$FILES; \
	fi

clean:
	@rm -rf .venv htmlcov/
	@rm -f .coverage coverage.xml
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
