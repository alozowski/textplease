.PHONY: style check all quality typecheck lock-check

# Apply ruff fixes to $(file) or the whole repo if file= is unset
style:
	uv run ruff format $(or $(file),.)
	uv run ruff check --fix $(or $(file),.)

# Check without applying fixes — mirrors lint.yml
check:
	uv run ruff format --check $(or $(file),.)
	uv run ruff check $(or $(file),.)

# Run ty type checker — mirrors typecheck.yml
typecheck:
	uvx ty check textplease/

# Verify uv.lock is up to date — mirrors uv-lock-check.yml
lock-check:
	uv lock --check

# Fix the entire codebase
all:
	$(MAKE) style

# Full CI-equivalent check: lint + types + lock
quality:
	$(MAKE) check
	$(MAKE) typecheck
	$(MAKE) lock-check
