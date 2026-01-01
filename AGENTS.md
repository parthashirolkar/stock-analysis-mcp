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

### Known Issues & Fixes

**Issue #1: pmdarima Resid Callable Error**
- **Symptom:** `complex() first argument must be a string or a number, not 'method'` when running `arima_model_diagnostics` with `auto_select=True`
- **Root Cause:** pmdarima ARIMA models have `resid` as a callable method, not an array. Code tried to create pandas Series from method reference: `pd.Series(self.model.resid)`
- **Fix Location:** `src/model_training.py:904-916` in `comprehensive_diagnostics()`
- **Solution:** Check if resid is callable and handle both cases:
  ```python
  if hasattr(self.model, "resid"):
      if callable(self.model.resid):
          # pmdarima: call method to get residuals
          residuals = pd.Series(self.model.resid())
      else:
          # statsmodels: resid is an array attribute
          residuals = pd.Series(self.model.resid)
  ```
- **Lesson Learned:** When accessing model attributes, always check if they're callable before using them as data. Different libraries (pmdarima vs statsmodels) may have different interfaces.

**Issue #2: Box-Cox Unpacking Error**
- **Symptom:** `not enough values to unpack (expected 3, got 2)` when using `transform="boxcox"` parameter
- **Root Cause:** scipy's `boxcox()` function returns 2 values: `(transformed_data, lambda_param)`, not 3
- **Fix Location:** `src/model_training.py:327` in `_apply_transformation()`
- **Solution:** Correct unpacking statement:
  ```python
  # Before (WRONG):
  transformed, lambda_param, _ = boxcox(shifted_data.values)
  
  # After (CORRECT):
  transformed, lambda_param = boxcox(shifted_data.values)
  ```
- **Lesson Learned:** Always verify function return signatures from library documentation. scipy's boxcox returns only 2 values in current versions.

**Issue #3: mle_retvals Attribute Error**
- **Symptom:** `'ARIMA' object has no attribute 'mle_retvals'` when running `forecast_arima_model` with `auto_select=True` (default)
- **Root Cause:** pmdarima's `auto_arima()` creates pmdarima wrapper models that don't expose `mle_retvals` directly. They store underlying statsmodels results in `arima_res_` attribute. The code was accessing `model.mle_retvals` which works for statsmodels models but fails for pmdarima models.
- **Fix Location:** Added `_get_model_convergence()` helper method in `src/model_training.py:1112-1135` and updated `server.py:1304`
- **Solution:** Use helper method that checks both model types:
  ```python
  def _get_model_convergence(self, model):
      try:
          # Try direct access (statsmodels)
          if hasattr(model, "mle_retvals"):
              return model.mle_retvals is not None
          # Try wrapped access (pmdarima stores statsmodels results in arima_res_)
          elif hasattr(model, "arima_res_") and hasattr(model.arima_res_, "mle_retvals"):
              return model.arima_res_.mle_retvals is not None
          # Fallback - assume converged if model exists
          return True
      except Exception:
          return True
  ```
  Then replace direct access with helper: `trainer._get_model_convergence(model)`
- **Lesson Learned:** pmdarima is a wrapper around statsmodels with a different interface. Always use `hasattr()` checks or create helper methods to handle both library interfaces safely. When working with multiple libraries that provide similar functionality, test both paths to ensure compatibility.
- **Additional Context:** pmdarima models store the underlying statsmodels ARIMAResults object in the `arima_res_` attribute. Accessing attributes through this wrapper requires checking which library's interface is available.

**Issue #4: Transformation Scale Mismatch in Plots**
- **Symptom:** When `transform="log"` or `transform="boxcox"` is used, historical data appears as a flat line near 0 while forecast values are at normal price scale (e.g., ₹3000).
- **Root Cause:** In `train_model()`, `self.data` is overwritten with transformed data while original data is stored in `self.original_data`. Plotting code was using `trainer.data` for historical data (transformed scale) but forecasts were already inverse-transformed back to original scale, causing scale mismatch.
- **Fix Locations:**
  - `src/model_training.py:335-380` - Fixed `_inverse_transform()` to handle lists, not just arrays and Series
  - `server.py:1184-1191` - Use `trainer.original_data` instead of `trainer.data` for historical plotting when transformation is applied
  - `server.py:1229` - Use `trainer.original_data` for last price line
  - `server.py:663-712` - Use `trainer.original_data` for train/test split plotting and inverse-transform fittedvalues when transformation is applied
- **Solution:**
  ```python
  # For plotting historical data
  if transform:
      historical_data = trainer.original_data
  else:
      historical_data = trainer.data

  # For plotting fittedvalues (train_arima_model)
  if transform:
      if callable(model.fittedvalues):
          fitted_vals = model.fittedvalues()
      else:
          fitted_vals = model.fittedvalues
      model_fitted = trainer._inverse_transform(fitted_vals[:len(train_data_plot)])
  else:
      if callable(model.fittedvalues):
          model_fitted = model.fittedvalues()
      else:
          model_fitted = model.fittedvalues
      model_fitted = model_fitted[:len(train_data_plot)]

  # For handling different input types in _inverse_transform
  if isinstance(data, np.ndarray):
      return np.exp(data)  # or other transformation
  elif isinstance(data, pd.Series):
      return np.exp(data.values)
  else:
      return np.exp(np.array(data))
  ```
- **Lesson Learned:** When data transformations are applied, `self.data` holds transformed data while `self.original_data` holds original scale data. Always use `original_data` for plotting and ensure forecasts (which are inverse-transformed) match the same scale. The `_inverse_transform()` method must handle multiple input types (arrays, Series, lists).
- **Additional Context:** This affects both `train_arima_model` and `forecast_arima_model` tools. The fix ensures all plotted data (historical, fitted, test, forecast) are on the same price scale regardless of transformation type.

**Prevention:**
- Test with different library versions during development
- Check return values before unpacking in tuple assignments
- Use try-except blocks around external library calls with descriptive error messages
- Document library-specific behavior in docstrings
- When using wrapper libraries (like pmdarima wrapping statsmodels), always check which interface is available before accessing attributes
- Create helper methods to abstract away library-specific differences when code must support multiple libraries
- When implementing transformations, track both original and transformed data separately and use the appropriate one based on context (model training vs plotting/visualization)
- Ensure utility methods like `_inverse_transform()` handle all possible input types (arrays, Series, lists) to avoid AttributeError
