# AGENTS.md - Agent Instructions for stock-analysis-mcp

This document provides guidelines for agentic coding tools working on this repository.

## Build/Lint/Test Commands

```bash
# Install dependencies
uv sync

# Run the MCP server
uv run python server.py

# Lint code
uv run ruff check .

# Format code
uv run ruff format .

# Type checking (if mypy is added)
uv run mypy .

```

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
- Use built-in types for simple values: `str`, `int`, `float`, `bool`
- Use `typing` module for complex types: `Dict[str, Any]`, `List[Dict[str, Any]]`, `Optional[str]`, `Tuple[...]`
- Async functions must be typed: `async def get_stock_quote(ticker: str) -> Dict[str, Any]:`
- Class methods should include `self: ARIMATrainer` type hints

### Naming Conventions
- **Functions:** `snake_case` (e.g., `get_stock_quote`, `normalize_ticker`, `analyze_acf`)
- **Classes:** `PascalCase` (e.g., `ARIMATrainer`, `SARIMATrainer`)
- **Variables:** `snake_case` (e.g., `stock_data`, `ticker_variants`, `current_price`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `DEFAULT_PERIOD`, `MAX_LAGS`)
- **Private methods:** prefix with underscore: `_encode_image`, `_load_data`

### Error Handling
- Always wrap external API calls (yfinance, etc.) in try-except blocks
- Raise descriptive exceptions: `ValueError(f"No data found for ticker {ticker}")`
- In tool functions, catch exceptions and return error messages for MCP compatibility
- Use specific exception types: `ValueError`, `RuntimeError`, `RuntimeError` vs generic `Exception`
- For MCP tools: `try: ... except Exception as e: return f"Error: {str(e)}"`
- For library functions: raise exceptions to let callers handle them

### Async/Await
- All data fetching functions must be async: `async def get_stock_quote(...)`
- Use `await` when calling async functions
- External blocking calls (yfinance) don't need async, but wrap them in async functions
- MCP tools must be async functions decorated with `@mcp.tool()`

### Docstrings
- **Format:** PEP 257 style with Args and Returns sections
- **Required** for all public functions and classes
- Include parameter types in descriptions (e.g., `ticker (str): Stock ticker symbol`)
- Return types should be documented: `Returns: Dict[str, Any] with stock data`
- Example:
```python
async def get_stock_quote(ticker: str) -> Dict[str, Any]:
    """
    Get current stock price and basic trading information

    Args:
        ticker: Indian stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')

    Returns:
        Dictionary with current stock price and trading data
    """
```

### Project Structure
```
stock-analysis-mcp/
├── server.py              # Main MCP server with FastMCP tools (@mcp.tool decorators)
├── src/
│   ├── __init__.py
│   ├── stock_analyzer.py  # Core data fetching (yfinance wrapper functions)
│   └── model_training.py  # ARIMA/SARIMA model training classes
├── pyproject.toml         # Dependencies and project config
├── Dockerfile            # Container build configuration
└── README.md             # User-facing documentation
```

### Architecture Patterns

**Tool Functions (server.py):**
- Decorated with `@mcp.tool()` for MCP discovery
- Always async functions returning `str` or `list` (for text + images)
- Format output with emoji sections for readability
- Handle errors gracefully: `return f"Error: {str(e)}"`

**Data Functions (src/stock_analyzer.py):**
- Pure async functions that fetch data from yfinance
- Return structured dicts with consistent keys
- Use `normalize_ticker()` to handle NSE/BSE variants
- Validate inputs and raise exceptions on errors

**Model Classes (src/model_training.py):**
- Class-based design: `ARIMATrainer`, `SARIMATrainer`
- Separate methods for: data loading, parameter selection, training, forecasting
- Store model instance in `self.model` after training
- Comprehensive docstrings for all public methods

### Logging
- **Level:** ERROR only (configured in server.py to avoid polluting MCP stdio)
- **Format:** `%(asctime)s - %(levelname)s - %(message)s`
- **Output:** stderr (MCP-safe)
- **Usage:** `logger.error(f"Error message")` sparingly

### Adding New Features
1. Add data fetching logic to `src/stock_analyzer.py`
2. Create MCP tool in `server.py` with `@mcp.tool()` decorator
3. Follow existing patterns for error handling and output formatting
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
- Type safety: use `.get()` with defaults for dict access to avoid KeyError

### MCP-Specific Guidelines
- Tools return `str` for text-only responses
- Tools returning images should return `list[str, ImageContent]`
- Use base64 encoding for images via `_encode_image()` helper
- Avoid stdout output (MCP uses stdio for communication)
- Log to stderr only in ERROR mode
- Tool descriptions should be user-friendly and comprehensive
