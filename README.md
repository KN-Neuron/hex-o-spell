# Hex-O-Spell: EEG Classification Project

This project implements EEG-based classification using PyTorch and scikit-learn.

## Features

- EEG data preprocessing pipeline
- Neural network classification models
- Standardized preprocessing with scikit-learn
- PyTorch-based deep learning models
- Comprehensive testing suite

## Installation

1. Clone the repository
2. Install the dependencies using Poetry:

```bash
poetry install
```

To install development dependencies as well:

```bash
poetry install --with dev
```

## Usage

Run the main training script:

```bash
poetry run python src/train.py
```

## Testing

This project includes a comprehensive testing suite using pytest. To run the tests:

1. Run all tests:

```bash
poetry run pytest
```

2. Run tests with coverage:

```bash
poetry run pytest --cov=src --cov-report=html
```

3. Run specific test files:

```bash
poetry run pytest tests/test_models.py
poetry run pytest tests/test_preprocessing.py
poetry run pytest tests/test_integration.py
```

4. Run tests with verbose output:

```bash
poetry run pytest -v
```

### Test Structure

- `tests/test_basic.py`: Basic functionality tests
- `tests/test_models.py`: Tests for the EEG classification models
- `tests/test_preprocessing.py`: Tests for the EEG preprocessing pipeline
- `tests/test_integration.py`: Integration tests for the full pipeline
- `tests/conftest.py`: Test fixtures and configuration

### Code Quality

The project uses the following tools for code quality:

- `black` for code formatting
- `flake8` for linting
- `mypy` for static type checking

Run code formatting:

```bash
poetry run black src/ tests/
```

Run linting:

```bash
poetry run flake8 src/ tests/
```

Run type checking:

```bash
poetry run mypy src/
```
