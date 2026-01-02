# AGENTS.md - Agent Instructions for stock-analysis-mcp

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
- **Classes:** `PascalCase` (e.g., `ARIMATrainer`, `ProphetTrainer`)
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

### Architecture Patterns
**Tool Functions (server.py):** Async with `@mcp.tool()`, return `str` or `list[str, ImageContent]`, format with emoji, handle errors
**Data Functions (src/stock_analyzer.py):** Pure async functions, structured dicts, use `normalize_ticker()` for NSE/BSE variants
**Model Classes (src/model_training.py):** Class-based (ARIMATrainer, SARIMATrainer, ProphetTrainer), separate methods for load/train/forecast, **Prophet-specific:** use `freq="B"`, normalize dates, support Indian holidays

### Logging
- **Level:** ERROR only (configured in server.py)
- **Format:** `%(asctime)s - %(levelname)s - %(message)s`
- **Output:** stderr (MCP-safe)
- **Usage:** `logger.error(f"Error message")` sparingly

### Adding New Features
1. Add data fetching logic to `src/stock_analyzer.py`
2. For forecasting models: Add class to `src/model_training.py`
3. Create MCP tool in `server.py` with `@mcp.tool()` decorator
4. Follow error handling and output formatting patterns
5. Run `uv run ruff check .` before committing

### Dependencies
- **Manager:** `uv` (not pip)
- **Lock file:** `uv.lock` (commit this!)
- **Add deps:** Edit `pyproject.toml`, run `uv sync`
- **Key deps:** FastMCP, yfinance, pandas, matplotlib, statsmodels, scikit-learn, prophet

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

## Debugging Strategy

**ALWAYS use bash tool with `uv run` to debug - never write test scripts**

When debugging code issues:
1. Use the Bash tool to run Python scripts directly via `uv run python -c "..."` or `uv run python <script.py>`
2. Get immediate traceback errors as feedback
3. Avoid creating test files that create garbage code and require cleanup

**Example:**
```bash
# Good - Direct execution with traceback
uv run python -c "from src.model_training import ProphetTrainer; trainer = ProphetTrainer('TCS', '1y'); print(trainer.load_data())"

# Bad - Creates test.py file that needs cleanup
# (Don't write test.py files for debugging)
```

This approach provides immediate error feedback without polluting the codebase with test files.

## Critical Patterns to Avoid Bugs

**Library Interface Compatibility:**
- pmdarima vs statsmodels: Always check `hasattr()` before accessing model attributes like `resid`, `mle_retvals`
- Create helper methods to abstract library differences (e.g., `_get_model_convergence()`)

**Data Transformations:**
- Store both `self.data` (transformed) and `self.original_data` (original scale)
- Use `original_data` for plotting when transformations are applied
- Ensure `_inverse_transform()` handles multiple input types (np.ndarray, pd.Series, list)

**Matplotlib Compatibility:**
- Convert pandas DatetimeIndex to NumPy array: `.to_numpy()` or `.values` before plotting
- Always call `plt.close()` and `buf.close()` after saving figures

**Prophet-Specific:**
- Use business-day frequency: `freq="B"` for trading data
- Normalize dates: `pd.to_datetime().normalize()` for Prophet compatibility
- Use `model.make_future_dataframe(periods=N, freq="B")` instead of date matching
