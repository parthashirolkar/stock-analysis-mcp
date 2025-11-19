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
    get_technical_indicators,
    get_stock_actions,
    get_analyst_recommendations,
    get_stock_holders,
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
                        published_date = parser.parse(item["published"]).strftime(
                            "%Y-%m-%d"
                        )
                    else:
                        # Fallback to timestamp
                        published_date = datetime.fromtimestamp(
                            item["published"]
                        ).strftime("%Y-%m-%d")
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
            df = df.set_index("date")
            # Reorder columns to match OHLCV format
            df = df[["open", "high", "low", "close", "volume"]]
            result += df.to_markdown(floatfmt=".2f")
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

    return ImageContent(type="image", data=img_base64, mimeType="image/png")


@mcp.tool()
async def bollinger_bands(
    symbol: str, period: str = "3mo", interval: str = "1d"
) -> list:
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
        analysis_text = f"""BOLLINGER BANDS ANALYSIS - {analysis["symbol"]}

📊 Price Information:
• Current Price: Rs.{analysis["current_price"]}
• Price Range: Rs.{analysis["price_range"]["min"]} - Rs.{analysis["price_range"]["max"]}
• Data Points: {analysis["data_points"]} ({analysis["date_range"]["start"]} to {analysis["date_range"]["end"]})

📈 Position Analysis:
• Band Position: {analysis["position_percentage"]}% ({analysis["position_status"]})"""

        if analysis["position_percentage"] > 75:
            analysis_text += (
                "\n⚠️ Stock is approaching OVERBOUGHT levels - Consider taking profits"
            )
        elif analysis["position_percentage"] < 25:
            analysis_text += "\n✅ Stock is approaching OVERSOLD levels - Potential buying opportunity"
        else:
            analysis_text += "\n📊 Stock is in NEUTRAL territory"

        analysis_text += f"""

📉 Volatility Analysis:
• Current Band Width: {analysis["current_band_width"]}%
• Average Band Width: {analysis["average_band_width"]}%
• Volatility Status: {analysis["volatility_status"]}"""

        if analysis["squeeze_detected"]:
            analysis_text += "\n🔔 BAND SQUEEZE DETECTED! Low volatility period - Major move likely coming soon"

        analysis_text += f"""

📋 Technical Summary:
• Analysis Period: {analysis["period"]} ({analysis["interval"]})
• Current Trend: {"UPWARD" if analysis["current_price"] > analysis["price_range"]["min"] + (analysis["price_range"]["max"] - analysis["price_range"]["min"]) * 0.5 else "SIDEWAYS/DOWNWARD"}
• Trading Range: {"WIDE" if analysis["current_band_width"] > analysis["average_band_width"] * 1.1 else "NARROW"}"""

        # Encode the PIL Image as MCP ImageContent
        image_content = _encode_image(chart_image)

        return [analysis_text.strip(), image_content]

    except Exception as e:
        # Create an error image instead of just returning text
        error_image = PILImage.new("RGB", (600, 200), color="red")
        error_content = _encode_image(error_image)

        return [f"Error generating Bollinger Bands analysis: {str(e)}", error_content]


@mcp.tool()
async def technical_analysis(ticker: str, period: str = "3mo") -> str:
    """Get comprehensive technical analysis indicators for an Indian stock.

    Provides key technical indicators including:
    - RSI (Relative Strength Index) for momentum
    - MACD (Moving Average Convergence Divergence) for trend signals
    - SMA/EMA (Simple/Exponential Moving Averages) for trend direction
    - Rate of Change for momentum analysis
    - Bollinger Bands for volatility and price position

    Args:
        ticker: Indian stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
        period: Time period for analysis (1mo, 3mo, 6mo, 1y, 2y)
    """
    try:
        data = await get_technical_indicators(ticker, period)
        indicators = data["indicators"]

        result = f"""TECHNICAL ANALYSIS - {data["ticker"]} ({data["exchange"]})
Current Price: ₹{data["current_price"]:.2f}
Analysis Period: {data["period"]} ({data["data_points"]} data points)

🔄 MOMENTUM INDICATORS:

📈 RSI (14-day): {indicators["RSI"]["current"]}
Status: {indicators["RSI"]["status"]}"""

        if indicators["RSI"]["status"] == "Overbought":
            result += "\n⚠️ Stock is overbought - Consider taking profits or wait for correction"
        elif indicators["RSI"]["status"] == "Oversold":
            result += "\n✅ Stock is oversold - Potential buying opportunity"

        result += """

📊 MOVING AVERAGES:

SMA Analysis:"""

        if indicators["SMA"]["SMA_50"]:
            result += f"""
• 50-day SMA: ₹{indicators["SMA"]["SMA_50"]:.2f}
• 50-day Position: Price is {indicators["SMA"]["price_vs_sma50"]} SMA"""

        if indicators["SMA"]["SMA_200"]:
            result += f"""
• 200-day SMA: ₹{indicators["SMA"]["SMA_200"]:.2f}
• 200-day Position: Price is {indicators["SMA"]["price_vs_sma200"]} SMA"""

        result += f"""

EMA Analysis:
• 12-day EMA: ₹{indicators["EMA"]["EMA_12"]:.2f}
• 26-day EMA: ₹{indicators["EMA"]["EMA_26"]:.2f}
• EMA Position: Price is {indicators["EMA"]["price_vs_ema12"]} 12-day EMA, {indicators["EMA"]["price_vs_ema26"]} 26-day EMA

📈 MACD SIGNALS:
• MACD Line: {indicators["MACD"]["MACD_line"]:.4f}
• Signal Line: {indicators["MACD"]["Signal_line"]:.4f}
• Histogram: {indicators["MACD"]["Histogram"]:.4f}
• Crossover Signal: {indicators["MACD"]["crossover_signal"]}"""

        if indicators["MACD"]["crossover_signal"] == "Bullish":
            result += "\n🟢 Bullish crossover detected - Potential upward momentum"
        elif indicators["MACD"]["crossover_signal"] == "Bearish":
            result += "\n🔴 Bearish crossover detected - Potential downward pressure"

        result += f"""

⚡ MOMENTUM CHECK:
• 5-day Rate of Change: {indicators["Momentum"]["ROC_5_days"]:.2f}% (if available)
• 20-day Rate of Change: {indicators["Momentum"]["ROC_20_days"]:.2f}% (if available)
• Momentum Status: {indicators["Momentum"]["momentum_status"]}

📊 BOLLINGER BANDS POSITION:
• Current Position: {indicators["Bollinger_Bands"]["position"]}% ({indicators["Bollinger_Bands"]["status"]})
• Upper Band: ₹{indicators["Bollinger_Bands"]["upper_band"]:.2f}
• Middle Band: ₹{indicators["Bollinger_Bands"]["middle_band"]:.2f}
• Lower Band: ₹{indicators["Bollinger_Bands"]["lower_band"]:.2f}

🔍 OVERALL TECHNICAL SUMMARY:"""

        # Calculate overall technical outlook
        bullish_signals = 0
        bearish_signals = 0

        if indicators["RSI"]["status"] == "Oversold":
            bullish_signals += 1
        elif indicators["RSI"]["status"] == "Overbought":
            bearish_signals += 1

        if indicators["SMA"].get("price_vs_sma50") == "Above":
            bullish_signals += 1
        elif indicators["SMA"].get("price_vs_sma50") == "Below":
            bearish_signals += 1

        if indicators["MACD"]["crossover_signal"] == "Bullish":
            bullish_signals += 1
        elif indicators["MACD"]["crossover_signal"] == "Bearish":
            bearish_signals += 1

        if indicators["Bollinger_Bands"]["position"] < 30:
            bullish_signals += 1
        elif indicators["Bollinger_Bands"]["position"] > 70:
            bearish_signals += 1

        if bullish_signals > bearish_signals:
            result += "\n🟢 BULLISH - More technical indicators suggest upward movement"
        elif bearish_signals > bullish_signals:
            result += (
                "\n🔴 BEARISH - More technical indicators suggest downward pressure"
            )
        else:
            result += "\n🟡 NEUTRAL - Technical indicators show mixed signals"

        result += f"\n\nAnalysis Timestamp: {data['analysis_timestamp']}"

        return result.strip()

    except Exception as e:
        return f"Error generating technical analysis for {ticker}: {str(e)}"


@mcp.tool()
async def stock_actions(ticker: str) -> str:
    """Get corporate actions (dividends, stock splits) for an Indian stock.

    Provides information about:
    - Recent dividend payments and yields
    - Dividend history and frequency
    - Stock split history and ratios
    - Dividend stability analysis

    Args:
        ticker: Indian stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
    """
    try:
        data = await get_stock_actions(ticker)
        dividend_info = data["dividend_info"]
        split_info = data["stock_split_info"]
        analysis = data["dividend_analysis"]

        result = f"""CORPORATE ACTIONS - {data["ticker"]} ({data["exchange"]})
Current Price: ₹{data["current_price"]:.2f}

💰 DIVIDEND INFORMATION:

📈 Current Yield: {dividend_info["dividend_yield_percent"]}%
• Trailing Annual Dividend: ₹{dividend_info["trailing_annual_dividend"]:.2f}
• Estimated Annual Dividend: ₹{dividend_info["estimated_annual_dividend"]:.2f}
• Dividend Frequency: {dividend_info["dividend_frequency"]}
• 5-Year Total Dividends: ₹{dividend_info["total_dividends_5years"]:.2f} ({dividend_info["dividend_count_5years"]} payments)

📅 Recent Dividend History:"""

        if dividend_info["recent_dividends"]:
            for i, dividend in enumerate(dividend_info["recent_dividends"][-5:], 1):
                result += f"\n{i}. {dividend['date']}: ₹{dividend['amount']:.2f}"
        else:
            result += "\n• No recent dividend payments"

        result += f"""

📊 DIVIDEND ANALYSIS:
• Status: {"Dividend Paying" if analysis["is_dividend_paying"] else "Non-Dividend Paying"}
• Stability: {analysis["dividend_stability"]}
• Yield Status: {analysis["current_yield_status"]}
• Estimated Annual Payout: ₹{analysis["payout_estimate_inr"]:.2f}"""

        if analysis["dividend_stability"] == "Stable":
            result += "\n✅ Consistent dividend payment history"
        elif analysis["dividend_stability"] == "Irregular":
            result += "\n⚠️ Irregular dividend payments"

        result += f"""

🔄 STOCK SPLIT HISTORY:
• Total Splits: {split_info["total_splits_count"]}"""

        if split_info["recent_splits"]:
            result += "\n\n📅 Split History:"
            for i, split in enumerate(split_info["recent_splits"], 1):
                result += (
                    f"\n{i}. {split['date']}: {split['ratio']} ({split['description']})"
                )

            result += f"\n• Last Split: {split_info['last_split_date'] if split_info['last_split_date'] else 'N/A'}"
            result += f"\n• Last Split Ratio: {split_info['last_split_ratio'] if split_info['last_split_ratio'] else 'N/A'}"
        else:
            result += "\n• No stock splits in available history"

        result += """

💡 INVESTOR INSIGHTS:"""

        if dividend_info["dividend_yield_percent"] > 3:
            result += f"\n✅ High dividend yield ({dividend_info['dividend_yield_percent']:.1f}%) - Good for income investors"
        elif dividend_info["dividend_yield_percent"] > 1:
            result += f"\n📊 Moderate dividend yield ({dividend_info['dividend_yield_percent']:.1f}%)"
        elif analysis["is_dividend_paying"]:
            result += f"\n📉 Low dividend yield ({dividend_info['dividend_yield_percent']:.1f}%) - Focus may be on growth"
        else:
            result += "\n📈 Non-dividend paying - Likely reinvesting in growth"

        if split_info["total_splits_count"] > 0:
            result += f"\n🔄 Company has split shares {split_info['total_splits_count']} time(s) - Often indicates growth"
        else:
            result += "\n📊 No share splits - Steady share count"

        result += f"\n\nLast Updated: {data['last_updated']}"

        return result.strip()

    except Exception as e:
        return f"Error fetching stock actions for {ticker}: {str(e)}"


@mcp.tool()
async def analyst_recommendations(ticker: str) -> str:
    """Get analyst recommendations and price targets for an Indian stock.

    Provides professional market sentiment including:
    - Current analyst consensus (Buy/Sell/Hold)
    - Price targets and upside potential
    - Number of analysts covering the stock
    - Recent recommendation changes

    Args:
        ticker: Indian stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
    """
    try:
        data = await get_analyst_recommendations(ticker)
        price_targets = data["price_targets"]
        recommendations = data["recommendations_breakdown"]
        analysis = data["analysis_summary"]

        result = f"""ANALYST RECOMMENDATIONS - {data["ticker"]} ({data["exchange"]})
Current Price: ₹{data["current_price"]:.2f}

🎯 CURRENT CONSENSUS:
• Current Recommendation: {data["current_recommendation"]}
• Consensus Score: {data["consensus_score"]}/5 (if available)
• Overall Consensus: {data["consensus_recommendation"]}
• Number of Analysts: {data["analyst_count"] if data["analyst_count"] else "N/A"}

📊 PRICE TARGETS:"""

        if price_targets["mean"]:
            result += f"""
• Mean Target: ₹{price_targets["mean"]:.2f}
• High Target: ₹{price_targets["high"]:.2f}
• Low Target: ₹{price_targets["low"]:.2f}
• Median Target: ₹{price_targets["median"]:.2f}
• Upside Potential: {price_targets.get("upside_potential_percent", 0):.2f}%
• Target Based Signal: {price_targets.get("recommendation", "N/A")}"""

            if price_targets.get("upside_potential_percent", 0) > 10:
                result += "\n🟢 Significant upside potential according to analysts"
            elif price_targets.get("upside_potential_percent", 0) < -10:
                result += "\n🔴 Analysts see potential downside risk"
        else:
            result += "\n• Price targets not available"

        result += f"""

📈 RECOMMENDATION BREAKDOWN:
• Strong Buy: {recommendations["strong_buy"]}
• Buy: {recommendations["buy"]}
• Hold: {recommendations["hold"]}
• Sell: {recommendations["sell"]}
• Strong Sell: {recommendations["strong_sell"]}
• Total Recommendations: {data["total_recommendations_count"]}"""

        if data["recommendations_trend"]:
            result += """

📅 RECENT RECOMMENDATION CHANGES:"""
            for rec in data["recommendations_trend"][-3:]:  # Show last 3
                result += f"""
• {rec["date"]}: {rec["firm"]} - {rec["recommendation"]} ({rec["action"]})"""

        result += """

🔍 ANALYSIS SUMMARY:"""

        if analysis["strong_points"]:
            result += "\n✅ Strong Points:"
            for point in analysis["strong_points"]:
                result += f"\n  • {point}"

        if analysis["caution_points"]:
            result += "\n⚠️ Points to Consider:"
            for point in analysis["caution_points"]:
                result += f"\n  • {point}"

        result += """

💡 INVESTOR INSIGHTS:"""

        if data["consensus_recommendation"] in ["Strong Buy", "Buy"]:
            result += f"\n🟢 Strong analyst confidence - {data['consensus_recommendation']} consensus"
        elif data["consensus_recommendation"] in ["Strong Sell", "Sell"]:
            result += f"\n🔴 Analyst caution advised - {data['consensus_recommendation']} consensus"
        else:
            result += f"\n🟡 Mixed analyst sentiment - {data['consensus_recommendation']} consensus"

        if data["analyst_count"] and data["analyst_count"] >= 10:
            result += f"\n📊 Well-covered stock with {data['analyst_count']}+ analyst opinions"
        elif data["analyst_count"]:
            result += (
                f"\n📈 Moderate coverage with {data['analyst_count']} analyst opinions"
            )
        else:
            result += "\n❓ Limited analyst coverage available"

        result += f"\n\nLast Updated: {data['last_updated']}"

        return result.strip()

    except Exception as e:
        return f"Error fetching analyst recommendations for {ticker}: {str(e)}"


@mcp.tool()
async def stock_holders(ticker: str) -> str:
    """Get major holders and institutional ownership information for an Indian stock.

    Provides insights into:
    - Major shareholder categories and percentages
    - Top institutional holders and their positions
    - Insider ownership and trading activity
    - Ownership concentration analysis

    Args:
        ticker: Indian stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
    """
    try:
        data = await get_stock_holders(ticker)
        ownership_summary = data["ownership_summary"]
        analysis = data["ownership_analysis"]
        insights = data["insights"]

        result = f"""STOCK OWNERSHIP ANALYSIS - {data["ticker"]} ({data["exchange"]})
Market Cap: ₹{data["market_cap"]:,}

👥 MAJOR HOLDERS BREAKDOWN:"""

        if data["major_holders"]:
            for holder in data["major_holders"]:
                result += f"\n• {holder['category']}: {holder['display_value']}"
        else:
            result += "\n• Major holder data not available"

        result += f"""

🏦 INSTITUTIONAL OWNERSHIP:
• Total Institutional Holding: {ownership_summary["total_institutional_holding_percent"]}%
• Number of Institutional Holders: {ownership_summary["number_of_institutional_holders"]}
• Top Institutional Holder: {ownership_summary["top_institutional_holder"] if ownership_summary["top_institutional_holder"] else "N/A"}"""

        if data["institutional_holders"]:
            result += "\n\n📊 Top Institutional Holders:"
            for i, holder in enumerate(data["institutional_holders"][:5], 1):
                result += f"\n{i}. {holder['holder']}"
                result += f"\n   • Shares: {holder['shares']:,}"
                result += f"\n   • Ownership: {holder['percentage_out']}"
                if holder["date_reported"] != "N/A":
                    result += f"\n   • Date Reported: {holder['date_reported']}"
                result += ""

        result += """

👔 INSIDER OWNERSHIP:"""

        if data["insider_holders"]:
            result += f"\n• Number of Insider Holders: {ownership_summary['number_of_insider_holders']}"
            result += "\n\n📊 Key Insider Holders:"
            for i, holder in enumerate(data["insider_holders"][:3], 1):
                result += f"\n{i}. {holder['holder']}"
                result += f"\n   • Position: {holder['position']}"
                result += f"\n   • Shares: {holder['shares']:,}"
                if holder["date_reported"] != "N/A":
                    result += f"\n   • Date Reported: {holder['date_reported']}"
                result += ""
        else:
            result += "\n• Insider ownership data not available"

        result += f"""

📈 OWNERSHIP ANALYSIS:
• Ownership Strength: {insights["ownership_strength"]}
• Investor Type: {insights["investor_type"]}
• Ownership Distribution: {"Diverse" if analysis["diverse_ownership"] else "Concentrated"}
• Institutional Confidence: {"High" if analysis["high_institutional_confidence"] else "Moderate" if analysis["moderate_institutional_confidence"] else "Limited"}
• Ownership Concentration: {"High concentration detected" if analysis["concentrated_ownership"] else "Well distributed"}"""

        if insights["key_investors"]:
            result += f"\n\n🏆 Key Institutional Investors: {', '.join(insights['key_investors'][:3])}"

        result += """

💡 INVESTMENT IMPLICATIONS:"""

        if analysis["high_institutional_confidence"]:
            result += "\n✅ High institutional confidence suggests professional investor trust"
        elif analysis["moderate_institutional_confidence"]:
            result += (
                "\n📊 Moderate institutional ownership - balanced investor interest"
            )
        else:
            result += "\n⚠️ Low institutional ownership - may indicate higher volatility"

        if analysis["diverse_ownership"]:
            result += "\n🔄 Diverse ownership base - reduced single-entity risk"
        else:
            result += "\n🎯 Concentrated ownership - watch major shareholder actions"

        if analysis["concentrated_ownership"]:
            result += "\n⚠️ Concentrated ownership - monitor major shareholder decisions"

        result += f"\n\nLast Updated: {data['last_updated']}"

        return result.strip()

    except Exception as e:
        return f"Error fetching stock holders data for {ticker}: {str(e)}"


def main():
    """Initialize and run the server"""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
