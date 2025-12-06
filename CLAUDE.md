# Development Instructions for mcp-imagen-server

## Post-Implementation Validation

After significant implementation changes, always run the following validation steps:

### 1. Run Unit Tests

```bash
uv run pytest tests/ -v
```

This will execute all unit tests to ensure the implementation doesn't break existing functionality.

To run with coverage:
```bash
uv run pytest tests/ -v --cov=src
```

### 2. Run Linting (Ruff)

```bash
uv run ruff check .
```

Check for code quality issues and style violations.

### 3. Run Formatting Check

```bash
uv run ruff format --check .
```

Verify code formatting compliance.

### 4. Auto-fix Issues (if needed)

```bash
# Auto-fix linting issues
uv run ruff check --fix .

# Auto-format code
uv run ruff format .
```

## CI/CD

The project uses GitHub Actions for continuous integration:
- Unit tests run on every push and pull request
- Linting checks run on every push and pull request
- Both must pass before merging changes
