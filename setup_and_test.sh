#!/bin/bash
# Setup and test script for Hex-O-Spell EEG Classification Project

echo "Setting up Hex-O-Spell EEG Classification Project..."

# Install dependencies
echo "Installing dependencies..."
poetry install --no-root

echo "Running code quality checks..."

# Run black formatting
echo "Running black code formatter..."
poetry run black --check src/ tests/ || {
    echo "Code needs formatting, running black..."
    poetry run black src/ tests/
}

# Run flake8 linter
echo "Running flake8 linter..."
poetry run flake8 src/ tests/ --max-line-length=88

# Run mypy type checking (if configured)
echo "Running mypy type checking..."
poetry run mypy src/ || echo "Type checking failed or not configured - continuing..."

echo "Running tests..."
poetry run pytest -v

echo "Running tests with coverage..."
poetry run pytest --cov=src --cov-report=html --cov-report=term

echo "Setup and testing complete!"
echo "To run tests in the future, simply use: poetry run pytest"
echo "To run with coverage: poetry run pytest --cov=src --cov-report=html"