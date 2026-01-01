# Indian Stock Analysis MCP Server

A comprehensive Model Context Protocol (MCP) server for analyzing Indian stocks listed on BSE and NSE exchanges. Built with FastMCP and powered by Yahoo Finance API.

## Features

- **Real-time Stock Data**: Current prices, daily changes, volume, and market cap
- **Fundamental Analysis**: P/E ratios, ROE, debt-to-equity, and other key metrics
- **Historical Data**: OHLC data for technical analysis across multiple timeframes
- **Stock Discovery**: Search Indian stocks by company name or ticker symbol
- **Market Overview**: NIFTY 50, SENSEX, and other major Indian indices
- **News Integration**: Recent news articles for sentiment analysis (handled by LLM)
- **Market Status**: Real-time market hours and trading status
- **Indian Market Optimized**: Specialized for NSE (.NS) and BSE (.BO) exchanges

## Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) (recommended package manager)
- Claude Desktop (for MCP integration)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd stock-analysis-mcp
```

### 2. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 3. Verify Installation

```bash
uv run python -c "import server; print('MCP server installed successfully')"
```

## Claude Desktop Configuration

### 1. Locate Configuration File

Find your Claude Desktop configuration file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### 2. Add MCP Server Configuration

```json
{
  "mcpServers": {
    "indian-stock-analysis": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/stock-analysis-mcp",
        "run",
        "server.py"
      ]
    }
  }
}
```

**Important**: Replace `/ABSOLUTE/PATH/TO/stock-analysis-mcp` with the actual absolute path to your project directory.

### 3. Restart Claude Desktop

Completely quit and restart Claude Desktop to load the new MCP server.

### 4. Verify Connection

Look for the tools icon in Claude Desktop. You should see the Indian stock analysis tools available.

## Available Tools

### 1. `stock_quote`
Get current stock price and basic trading information.

**Parameters:**
- `ticker` (required): Indian stock ticker symbol (e.g., "RELIANCE", "TCS", "INFY")

**Example Usage:**
```
What's the current price of Reliance Industries?
Show me the stock quote for TCS
```

**Sample Output:**
```
Current Stock Quote for RELIANCE (NSE):

Price Information:
* Current Price: ₹2,845.50
* Daily Change: +1.2% (₹33.80)
* Day's Range: ₹2,812.00 - ₹2,855.90
* Opening Price: ₹2,815.00
* Previous Close: ₹2,811.70

Trading Data:
* Volume: 15,234,567
* Market Cap: ₹19,82,123 Cr
* Currency: INR
```

### 2. `company_fundamentals`
Get comprehensive fundamental analysis data for a company.

**Parameters:**
- `ticker` (required): Indian stock ticker symbol

**Example Usage:**
```
Show me the fundamentals for Infosys
What are the financial metrics for HDFC Bank?
```

**Sample Output:**
```
Fundamental Analysis for INFOSYS LTD (INFY):

Company Information:
* Name: Infosys Limited
* Sector: Technology
* Industry: IT Services
* Exchange: NSE
* Website: https://www.infosys.com

Valuation Metrics:
* P/E Ratio: 28.5
* P/B Ratio: 8.2
* Market Cap: ₹6,45,789 Cr

Financial Metrics:
* EPS: ₹68.5
* ROE: 29.8%
* Dividend Yield: 2.1%
```

### 3. `stock_news`
Get recent news articles for a specific stock.

**Parameters:**
- `ticker` (required): Indian stock ticker symbol
- `limit` (optional): Maximum number of articles (default: 5)

**Example Usage:**
```
Show me recent news about Tata Motors
What are the latest news articles for ICICI Bank?
```

### 4. `search_indian_stocks`
Search for Indian stocks by company name or ticker symbol.

**Parameters:**
- `query` (required): Search query - company name or partial ticker
- `limit` (optional): Maximum number of results (default: 10)

**Example Usage:**
```
Search for banking stocks
Find companies with "tata" in the name
```

### 5. `market_overview`
Get current Indian market indices and sector performance.

**Example Usage:**
```
What's the current market overview?
Show me NIFTY and SENSEX performance
```

**Sample Output:**
```
Indian Market Overview:

Market Indices:
* NIFTY 50: ₹21,834.25 (+0.8%)
* SENSEX: ₹72,156.85 (+0.6%)
* NIFTY BANK: ₹46,789.30 (+1.2%)
* NIFTY IT: ₹29,456.20 (-0.3%)

Market Status: Open
Market is currently open for trading
Current Time: 2024-01-15 14:30:25 IST
```

## Available Resources

### 1. `indian-stock://market-status`
Current Indian market status and trading hours.

**Access through:** MCP client resource interface

### 2. `indian-stock://popular-stocks`
List of frequently analyzed Indian stocks with basic information.

**Access through:** MCP client resource interface

## Usage Examples

### Portfolio Analysis Workflow
```
User: "Show me fundamentals of Reliance Industries"
[Uses company_fundamentals tool]

User: "What's current market status?"
[Uses market_overview tool]

User: "Get recent news about Reliance"
[Uses stock_news tool]

User: "How has Reliance performed technically over the past 6 months?"
[Uses technical_analysis tool]
```

### Stock Discovery Workflow
```
User: "Search for technology companies"
[Uses search_indian_stocks tool]

User: "Show me the current price of TCS"
[Uses stock_quote tool]

User: "What are TCS's financial metrics?"
[Uses company_fundamentals tool]
```

### Market Research Workflow
```
User: "What's today's market overview?"
[Uses market_overview tool]

User: "Find banking stocks with good fundamentals"
[Uses search_indian_stocks tool + company_fundamentals tool]

User: "Show me technical analysis for top performers"
[Uses technical_analysis tool]
```

## Development

### Project Structure
```
stock-analysis-mcp/
├── server.py              # Main MCP server with FastMCP tools
├── stock_analyzer.py      # Core data fetching and processing
├── pyproject.toml         # Project configuration and dependencies
├── main.py               # Entry point
└── README.md             # This file
```

### Code Quality
- Uses **ruff** for linting and code formatting
- Follows Python type hints and async/await patterns
- Comprehensive error handling
- Structured logging to stderr (MCP-safe)

### Running the Server Locally
```bash
# Start the MCP server
uv run python server.py

# The server will listen for JSON-RPC messages on stdin/stdout
```

### Adding New Tools
1. Add the data fetching function to `stock_analyzer.py`
2. Create the tool function in `server.py` with `@mcp.tool()` decorator
3. Follow the existing patterns for error handling and return formatting

## Troubleshooting

### Common Issues

**Server not showing up in Claude Desktop**
- Verify the absolute path in `claude_desktop_config.json`
- Completely quit and restart Claude Desktop
- Check that dependencies are installed: `uv sync`

**No data found for a ticker**
- Ensure the ticker is valid (try with .NS or .BO suffix)
- Check if the market is open
- Verify the stock is listed on NSE or BSE

**Error fetching data**
- Check network connection
- Yahoo Finance API may have rate limits
- Try again after a few minutes

### Market Timing Considerations
- **Market Hours**: 9:15 AM - 3:30 PM IST, Monday to Friday
- **Pre-market**: Data may be limited before 9:15 AM
- **Weekends**: No real-time data available
- **Holidays**: Indian market holidays affect data availability

### Data Limitations
- Real-time data is subject to Yahoo Finance API limitations
- Some fundamental data may not be available for all stocks
- Historical data accuracy depends on Yahoo Finance data quality
- News article availability varies by stock and source


## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and ensure they pass linting: `uv run ruff check .`
4. Commit your changes: `git commit -m "feat: add new feature"`
5. Push to the branch: `git push origin feature-name`
6. Open a pull request

## Support

For issues and questions:
- Check the troubleshooting section above
- Open an issue on the repository
- Review the MCP documentation for Claude Desktop integration

---

**Disclaimer**: This tool provides financial information for educational purposes only. Not financial advice. Always consult with qualified financial professionals before making investment decisions.