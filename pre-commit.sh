#!/bin/bash
# Pre-commit hook script
# Runs linting, formatting, type checking, and tests

set -e

echo "Running pre-commit checks..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if commands exist
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        exit 1
    fi
}

# Run ruff check (linting) — --fix auto-fixes what it can
run_ruff_check() {
    echo -e "${YELLOW}Running ruff (linting)...${NC}"
    ruff check --fix src/ tests/ examples/ --output-format=concise
    echo -e "${GREEN}✓ Ruff check passed${NC}"
}

# Run ruff format (formatting) — formats in place (no --check)
run_ruff_format() {
    echo -e "${YELLOW}Running ruff (formatting)...${NC}"
    ruff format src/ tests/ examples/
    echo -e "${GREEN}✓ Ruff format passed${NC}"
}

# Run mypy (type checking) — src/ only; tests/examples have relaxed typing in pyproject.toml
# Uses uv run to ensure dev extra (openai, numpy, etc.) is available for optional imports.
run_mypy() {
    echo -e "${YELLOW}Running mypy (type checking)...${NC}"
    rm -rf .mypy_cache
    uv run python -m mypy src/
    echo -e "${GREEN}✓ Mypy check passed${NC}"
}

# Run pytest (all tests must pass)
run_pytest() {
    echo -e "${YELLOW}Running pytest (tests)...${NC}"
    uv run python -m pytest tests/ -q --tb=short
    echo -e "${GREEN}✓ All tests passed${NC}"
}

# Run pip-audit (CVE / vulnerability scan of installed dependencies)
run_pip_audit() {
    echo -e "${YELLOW}Running pip-audit (CVE scan)...${NC}"
    if ! uv run python -m pip_audit --version &> /dev/null 2>&1; then
        echo -e "${YELLOW}pip-audit not installed — skipping CVE scan (run: uv add --dev pip-audit)${NC}"
        return 0
    fi
    # CVE-2026-3219: pip ZIP/tar ambiguity — no fix released yet, ignored until patched
    uv run python -m pip_audit --skip-editable --desc on --ignore-vuln CVE-2026-3219 2>&1
    echo -e "${GREEN}✓ pip-audit passed${NC}"
}

# Run playground ESLint + Prettier
run_playground_checks() {
    if [ ! -d "playground" ]; then
        echo -e "${YELLOW}Skipping playground checks (no playground/ directory)${NC}"
        return 0
    fi
    if ! command -v npm &> /dev/null; then
        echo -e "${YELLOW}Skipping playground checks (npm not installed)${NC}"
        return 0
    fi
    echo -e "${YELLOW}Running playground ESLint...${NC}"
    (cd playground && npm ci --silent 2>/dev/null || npm install --silent) && \
    (cd playground && npm run lint)
    echo -e "${GREEN}✓ Playground ESLint passed${NC}"
    echo -e "${YELLOW}Running playground Prettier (format check)...${NC}"
    (cd playground && npm run format:check)
    echo -e "${GREEN}✓ Playground Prettier check passed${NC}"
}

# Main execution
echo ""
echo "========================================="
echo "Pre-commit Quality Checks"
echo "========================================="
echo ""

# Run checks
run_ruff_check
run_ruff_format
run_mypy
run_pytest
run_pip_audit
run_playground_checks

echo ""
echo "========================================="
echo -e "${GREEN}All pre-commit checks passed!${NC}"
echo "========================================="
