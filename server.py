#!/usr/bin/env python3
"""
Indian Stock Analysis MCP Server

An MCP server providing comprehensive Indian stock analysis tools
for BSE/NSE listed stocks using FastMCP.
"""

import logging
import io
import base64
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent
from PIL import Image as PILImage

# Import stock analysis functionality
from stock_analyzer import (
    get_stock_quote,
    get_company_fundamentals,
    get_stock_news,
    search_stocks,
    get_historical_data,
    get_market_overview,
    get_market_status,
    create_bollinger_chart,
    get_popular_stocks,
)

# Configure logging to stderr (safe for MCP)
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("indian-stock-mcp")

# Initialize FastMCP server
mcp = FastMCP("indian-stock-analysis")


@mcp.tool()
async def stock_quote(ticker: str) -> str:
    """Get current stock price and basic trading information for Indian stocks.

    Args:
        ticker: Indian stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
    """
    try:
        data = await get_stock_quote(ticker)
        return f"""
Current Stock Quote for {data["ticker"]} ({data["exchange"]}):

📈 Price Information:
• Current Price: ₹{data["current_price"]:.2f}
• Daily Change: {data["change"]:.2f}% (₹{data["change_amount"]:.2f})
• Day's Range: ₹{data["low"]:.2f} - ₹{data["high"]:.2f}
• Opening Price: ₹{data["open"]:.2f}
• Previous Close: ₹{data["previous_close"]:.2f}

📊 Trading Data:
• Volume: {data["volume"]:,}
• Market Cap: ₹{data["market_cap"]:,} Cr
• Currency: {data["currency"]}

Last Updated: {data["last_updated"]}
        """.strip()
    except Exception as e:
        return f"Error fetching stock quote for {ticker}: {str(e)}"


@mcp.tool()
async def company_fundamentals(ticker: str) -> str:
    """Get fundamental analysis data for an Indian company.

    Args:
        ticker: Indian stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
    """
    try:
        data = await get_company_fundamentals(ticker)
        return f"""
Fundamental Analysis for {data["company_name"]} ({data["ticker"]}):

🏢 Company Information:
• Name: {data["company_name"]}
• Sector: {data["sector"]}
• Industry: {data["industry"]}
• Exchange: {data["exchange"]}
• Website: {data["website"]}

📊 Valuation Metrics:
• P/E Ratio: {data["pe_ratio"] if data["pe_ratio"] else "N/A"}
• P/B Ratio: {data["pb_ratio"] if data["pb_ratio"] else "N/A"}
• Market Cap: ₹{data["market_cap"]:,} Cr

💰 Financial Metrics:
• EPS (Earnings per Share): ₹{data["eps"] if data["eps"] else "N/A"}
• Book Value: ₹{data["book_value"] if data["book_value"] else "N/A"}
• ROE (Return on Equity): {data["roe"] if data["roe"] else "N/A"}%
• Dividend Yield: {data["dividend_yield"] if data["dividend_yield"] else "N/A"}%

📈 Performance:
• 52-Week High: ₹{data["52_week_high"]:.2f}
• 52-Week Low: ₹{data["52_week_low"]:.2f}
• Average Volume: {data["avg_volume"]:,}
• Beta: {data["beta"] if data["beta"] else "N/A"}

🏦 Financial Health:
• Debt-to-Equity: {data["debt_to_equity"] if data["debt_to_equity"] else "N/A"}
• Current Ratio: {data["current_ratio"] if data["current_ratio"] else "N/A"}
• Profit Margin: {data["profit_margin"] if data["profit_margin"] else "N/A"}%
• Operating Margin: {data["operating_margin"] if data["operating_margin"] else "N/A"}%

Business Summary: {data["business_summary"]}...

Last Updated: {data["last_updated"]}
        """.strip()
    except Exception as e:
        return f"Error fetching fundamentals for {ticker}: {str(e)}"


@mcp.tool()
async def stock_news(ticker: str, limit: int = 5) -> str:
    """Get recent news articles for a specific Indian stock.

    Args:
        ticker: Indian stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
        limit: Maximum number of news articles to return (default: 5)
    """
    try:
        news_items = await get_stock_news(ticker, limit)

        if not news_items:
            return f"No recent news found for {ticker}"

        result = f"Recent News for {ticker}:\n\n"

        for i, item in enumerate(news_items, 1):
            published_date = ""
            if item.get("published"):
                try:
                    from datetime import datetime
                    from dateutil import parser

                    # Try to parse as ISO date string first
                    if isinstance(item["published"], str):
                        published_date = parser.parse(item["published"]).strftime("%Y-%m-%d")
                    else:
                        # Fallback to timestamp
                        published_date = datetime.fromtimestamp(item["published"]).strftime("%Y-%m-%d")
                except (ValueError, TypeError, OSError, ImportError):
                    # Fallback: try simple string parsing
                    try:
                        # Extract just the date part from ISO string
                        date_str = str(item["published"]).split("T")[0]
                        published_date = date_str
                    except (ValueError, IndexError, AttributeError):
                        published_date = "Unknown date"

            result += f"""
{i}. {item["title"]}
   Publisher: {item["publisher"]}
   Date: {published_date}
   Summary: {item["summary"]}...
   URL: {item["url"]}

---"""

        return result.strip()
    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"


@mcp.tool()
async def search_indian_stocks(query: str, limit: int = 10) -> str:
    """Search for Indian stocks by company name or ticker symbol.

    Args:
        query: Search query - company name or partial ticker symbol
        limit: Maximum number of results to return (default: 10)
    """
    try:
        results = await search_stocks(query, limit)

        if not results:
            return f"No stocks found matching '{query}'"

        result = f"Search Results for '{query}':\n\n"

        for i, stock in enumerate(results, 1):
            result += f"""
{i}. {stock["ticker"]} - {stock["name"]}
   Sector: {stock["sector"]}
---"""

        return result.strip()
    except Exception as e:
        return f"Error searching stocks for '{query}': {str(e)}"


@mcp.tool()
async def historical_data(ticker: str, period: str = "1M") -> str:
    """Get historical price data for technical analysis.

    Args:
        ticker: Indian stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
        period: Time period for historical data (1D, 1W, 1M, 3M, 6M, 1Y, 2Y, 5Y)
    """
    try:
        data = await get_historical_data(ticker, period)

        if not data or not data.get("data"):
            return f"No historical data found for {ticker}"

        total_data_points = len(data["data"])

        if total_data_points <= 10:
            result = f"Historical Data for {data['ticker']} ({data['period']}) - Last {total_data_points} data points:\n\n"
            recent_data = data["data"]
        else:
            result = f"Historical Data for {data['ticker']} ({data['period']}) - Showing last 10 of {total_data_points} data points:\n\n"
            recent_data = data["data"][-10:]

        # Convert to pandas DataFrame and use to_markdown()
        try:
            import pandas as pd
            df = pd.DataFrame(recent_data)
            # Set date as index for better display
            df = df.set_index('date')
            # Reorder columns to match OHLCV format
            df = df[['open', 'high', 'low', 'close', 'volume']]
            result += df.to_markdown(floatfmt='.2f')
        except ImportError:
            # Fallback to manual formatting if pandas is not available
            result += "Date       | Open    | High    | Low     | Close   | Volume\n"
            result += "-" * 65 + "\n"
            for item in recent_data:
                result += f"{item['date']} | {item['open']:<7.2f} | {item['high']:<7.2f} | {item['low']:<7.2f} | {item['close']:<7.2f} | {item['volume']:,}\n"

        # Calculate some basic stats
        closes = [item["close"] for item in data["data"]]
        if len(closes) > 1:
            change = closes[-1] - closes[0]
            change_pct = (change / closes[0]) * 100
            result += f"\nPeriod Performance: {change_pct:+.2f}% (₹{change:+.2f})"
            result += f"\nPeriod High: ₹{max(closes):.2f}"
            result += f"\nPeriod Low: ₹{min(closes):.2f}"

        result += f"\n\nLast Updated: {data['last_updated']}"

        return result.strip()
    except Exception as e:
        return f"Error fetching historical data for {ticker}: {str(e)}"


@mcp.tool()
async def market_overview() -> str:
    """Get current Indian market indices and sector performance."""
    try:
        data = await get_market_overview()

        result = "Indian Market Overview:\n\n"

        # Display indices
        result += "📈 Market Indices:\n"
        for index_data in data["indices"].values():
            if "error" in index_data:
                result += f"• {index_data['name']}: Error fetching data\n"
            else:
                change_symbol = "📈" if index_data["change"] >= 0 else "📉"
                result += f"• {index_data['name']}: ₹{index_data['current_value']:.2f} "
                result += f"({index_data['change']:+.2f}%) {change_symbol}\n"

        # Display market status
        status_data = data["market_status"]
        result += f"\n🕐 Market Status: {status_data['status']}"
        result += f"\n{status_data['description']}"
        result += f"\nCurrent Time: {status_data['current_time']}"

        result += f"\n\nLast Updated: {data['last_updated']}"

        return result.strip()
    except Exception as e:
        return f"Error fetching market overview: {str(e)}"


# Add resources
@mcp.resource("indian-stock://market-status")
async def market_status_resource() -> str:
    """Current Indian market status and trading hours."""
    try:
        status = await get_market_status()
        return f"""
Market Status: {status["status"]}
{status["description"]}
Current Time: {status["current_time"]}
Market Hours: {status["market_open"]} - {status["market_close"]} {status["timezone"]}
        """.strip()
    except Exception as e:
        return f"Error fetching market status: {str(e)}"


@mcp.resource("indian-stock://popular-stocks")
async def popular_stocks_resource() -> str:
    """List of frequently analyzed Indian stocks."""
    try:
        stocks = await get_popular_stocks()
        result = "Popular Indian Stocks:\n\n"

        for stock in stocks:
            result += f"• {stock['ticker']} - {stock['name']} ({stock['sector']})\n"
            result += f"  {stock['description']}\n\n"

        return result.strip()
    except Exception as e:
        return f"Error fetching popular stocks: {str(e)}"


def _encode_image(image) -> ImageContent:
    """Encodes a PIL Image to a format compatible with ImageContent."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()

    return ImageContent(
        type="image",
        data=img_base64,
        mimeType="image/png"
    )


@mcp.tool()
async def bollinger_bands(symbol: str, period: str = "3mo", interval: str = "1d") -> list:
    """
    Generate Bollinger Bands analysis with visual chart for an Indian stock.

    Returns a comprehensive Bollinger Bands analysis including:
    - Price chart with Bollinger Bands overlay
    - Volatility analysis (Band Width)
    - Current position analysis (overbought/oversold)
    - Band squeeze detection
    - Visual chart as MCP ImageContent

    Args:
        symbol: Stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
        period: Time period for analysis ('1mo', '3mo', '6mo', '1y', '2y')
        interval: Data interval ('1d' for daily, '1h' for hourly, '5m' for 5-minute)

    Returns:
        List containing text analysis and ImageContent with Bollinger Bands chart
    """
    try:
        # Generate Bollinger Bands chart (returns PIL Image and analysis)
        chart_image, analysis = create_bollinger_chart(symbol, period, interval)

        # Format analysis text
        analysis_text = f"""BOLLINGER BANDS ANALYSIS - {analysis['symbol']}

📊 Price Information:
• Current Price: Rs.{analysis['current_price']}
• Price Range: Rs.{analysis['price_range']['min']} - Rs.{analysis['price_range']['max']}
• Data Points: {analysis['data_points']} ({analysis['date_range']['start']} to {analysis['date_range']['end']})

📈 Position Analysis:
• Band Position: {analysis['position_percentage']}% ({analysis['position_status']})"""

        if analysis['position_percentage'] > 75:
            analysis_text += "\n⚠️ Stock is approaching OVERBOUGHT levels - Consider taking profits"
        elif analysis['position_percentage'] < 25:
            analysis_text += "\n✅ Stock is approaching OVERSOLD levels - Potential buying opportunity"
        else:
            analysis_text += "\n📊 Stock is in NEUTRAL territory"

        analysis_text += f"""

📉 Volatility Analysis:
• Current Band Width: {analysis['current_band_width']}%
• Average Band Width: {analysis['average_band_width']}%
• Volatility Status: {analysis['volatility_status']}"""

        if analysis['squeeze_detected']:
            analysis_text += "\n🔔 BAND SQUEEZE DETECTED! Low volatility period - Major move likely coming soon"

        analysis_text += f"""

📋 Technical Summary:
• Analysis Period: {analysis['period']} ({analysis['interval']})
• Current Trend: {'UPWARD' if analysis['current_price'] > analysis['price_range']['min'] + (analysis['price_range']['max'] - analysis['price_range']['min']) * 0.5 else 'SIDEWAYS/DOWNWARD'}
• Trading Range: {'WIDE' if analysis['current_band_width'] > analysis['average_band_width'] * 1.1 else 'NARROW'}"""

        # Encode the PIL Image as MCP ImageContent
        image_content = _encode_image(chart_image)

        return [analysis_text.strip(), image_content]

    except Exception as e:
        # Create an error image instead of just returning text
        error_image = PILImage.new("RGB", (600, 200), color="red")
        error_content = _encode_image(error_image)

        return [f"Error generating Bollinger Bands analysis: {str(e)}", error_content]


def main():
    """Initialize and run the server"""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
