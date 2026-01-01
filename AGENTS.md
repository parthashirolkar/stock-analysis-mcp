# AGENTS.md - Agent Instructions for stock-analysis-mcp

This document provides guidelines for agentic coding tools working on this repository.

## Build/Lint/Test Commands

```bash
# Install dependencies
uv sync

# Run the MCP server (use MCP Inspector or Claude Desktop for testing)
uv run python server.py

# Lint code
uv run ruff check .

# Format code
uv run ruff format .

# Type checking (if mypy is added)
uv run mypy .

# Run linter on specific file
uv run ruff check server.py
```

**Testing:** No automated test framework configured. Test tools via MCP Inspector or Claude Desktop.

## Code Style Guidelines

### Imports
- **Standard library** imports first (e.g., `import logging`, `from datetime import datetime`)
- **Third-party** imports second (e.g., `import yfinance as yf`, `import pandas as pd`)
- **Local imports** third (e.g., `from src.stock_analyzer import get_stock_quote`)
- Always use absolute imports for local modules: `from src.` prefix required

### Formatting
- **Tool:** `ruff` for both linting and formatting
- **Indentation:** 4 spaces
- **Line length:** No strict limit, but prefer readability under 120 chars
- **Spacing:** Follow PEP 8 standard spacing rules
- Run `uv run ruff format .` before committing

### Type Hints
- **Mandatory** for all function parameters and return values
- Use built-in types: `str`, `int`, `float`, `bool`, `None`
- Use `typing` module for complex types: `Dict[str, Any]`, `List[str]`, `Optional[str]`
- Async functions: `async def get_stock_quote(ticker: str) -> Dict[str, Any]:`
- Class methods: include `self: ARIMATrainer` type hints

### Naming Conventions
- **Functions:** `snake_case` (e.g., `get_stock_quote`, `normalize_ticker`)
- **Classes:** `PascalCase` (e.g., `ARIMATrainer`, `SARIMATrainer`)
- **Variables:** `snake_case` (e.g., `stock_data`, `ticker_variants`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `DEFAULT_PERIOD`, `MAX_LAGS`)
- **Private methods:** prefix with underscore: `_encode_image`, `_load_data`

### Error Handling
- Wrap external API calls (yfinance) in try-except blocks
- Raise descriptive exceptions: `ValueError(f"No data found for ticker {ticker}")`
- In tool functions, catch and return error messages: `return f"Error: {str(e)}"`
- For library functions: raise exceptions to let callers handle them
- Use specific exceptions: `ValueError`, `RuntimeError` vs generic `Exception`

### Async/Await
- All data fetching functions must be async: `async def get_stock_quote(...)`
- Use `await` when calling async functions
- External blocking calls don't need async, but wrap them in async functions
- MCP tools must be async with `@mcp.tool()` decorator

### Docstrings
- **Format:** PEP 257 style with Args and Returns sections
- **Required** for all public functions and classes
- Include parameter types: `ticker (str): Stock ticker symbol`
- Document return values: `Returns: Dict[str, Any] with stock data`

### Project Structure
```
stock-analysis-mcp/
├── server.py              # MCP server with FastMCP tools (@mcp.tool decorators)
├── src/
│   ├── __init__.py
│   ├── stock_analyzer.py  # Data fetching (yfinance wrapper functions)
│   └── model_training.py  # ARIMA/SARIMA model training classes
├── pyproject.toml         # Dependencies and config
└── Dockerfile            # Container build config
```

### Architecture Patterns

**Tool Functions (server.py):**
- Decorated with `@mcp.tool()` for MCP discovery
- Always async functions returning `str` or `list[str, ImageContent]`
- Format output with emoji sections
- Handle errors: `return f"Error: {str(e)}"`

**Data Functions (src/stock_analyzer.py):**
- Pure async functions fetching from yfinance
- Return structured dicts with consistent keys
- Use `normalize_ticker()` for NSE/BSE variants
- Validate inputs and raise exceptions on errors

**Model Classes (src/model_training.py):**
- Class-based design: `ARIMATrainer`, `SARIMATrainer`
- Separate methods: data loading, parameter selection, training, forecasting
- Store model instance in `self.model` after training
- Comprehensive docstrings for all public methods

### Logging
- **Level:** ERROR only (configured in server.py)
- **Format:** `%(asctime)s - %(levelname)s - %(message)s`
- **Output:** stderr (MCP-safe)
- **Usage:** `logger.error(f"Error message")` sparingly

### Adding New Features
1. Add data fetching logic to `src/stock_analyzer.py`
2. Create MCP tool in `server.py` with `@mcp.tool()` decorator
3. Follow error handling and output formatting patterns
4. Run `uv run ruff check .` before committing

### Dependencies
- **Manager:** `uv` (not pip)
- **Lock file:** `uv.lock` (commit this!)
- **Add deps:** Edit `pyproject.toml`, run `uv sync`
- **Key deps:** FastMCP, yfinance, pandas, matplotlib, statsmodels, scikit-learn

### Code Quality
- Defensive programming: validate inputs, check for empty data
- Fallback mechanisms: try multiple ticker variants (NSE/BSE)
- Cache models where appropriate (see `get_cached_model()` in ARIMATrainer)
- Resource cleanup: close matplotlib figures, close buffers
- Type safety: use `.get()` with defaults for dict access

### MCP-Specific Guidelines
- Tools return `str` for text-only responses
- Image tools return `list[str, ImageContent]`
- Use base64 encoding via `_encode_image()` helper
- Avoid stdout output (MCP uses stdio)
- Log to stderr only in ERROR mode
- Tool descriptions should be user-friendly
