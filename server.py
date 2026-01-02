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
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import stock analysis functionality
from src.stock_analyzer import (
    get_stock_quote,
    get_company_fundamentals,
    get_stock_news,
    search_stocks,
    get_market_overview,
    get_market_status,
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
async def train_arima_model(
    ticker: str,
    p: int = 1,
    d: int = 1,
    q: int = 1,
    validation_split: float = 0.2,
    auto_select: bool = True,
    lags: int = 40,
    period: str = "1y",
    transform: str = None,
) -> list:
    """
    Train ARIMA model with intelligent parameter selection using pmdarima auto_arima.

    Provides:
    - Model training with automated ARIMA order selection via pmdarima
    - Data transformation support (log, Box-Cox) for improved normality
    - Performance metrics and validation on holdout set
    - Model persistence with caching capability
    - Error handling and graceful fallbacks

    Args:
        ticker: Stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
        p: AR order (None for auto-selection)
        d: Differencing order (default 1 for stock prices)
        q: MA order (None for auto-selection)
        validation_split: Train-validation split ratio (default 0.2)
        auto_select: Use pmdarima auto_arima for parameter selection (True) or manual (False)
        lags: Number of lags for analysis (default 40, used for fallback ACF/PACF)
        period: Time period for training data ('1mo', '3mo', '6mo', '1y', '2y', '5y')
        transform: Data transformation type ("log", "boxcox", or None for no transformation)

    Returns:
        List containing text analysis and ImageContent with training plot
    """
    try:
        from src.model_training import ARIMATrainer

        # Initialize trainer
        trainer = ARIMATrainer(ticker, period)

        # Let train_model handle everything including data loading and auto-selection
        result = trainer.train_model(
            p=p if not auto_select else None,
            d=d,
            q=q if not auto_select else None,
            validation_split=validation_split,
            auto_select=auto_select,
            transform=transform,
        )

        try:
            # Create training plot
            buf = io.BytesIO()
            plt.figure(figsize=(14, 8))

            # Get train/test split for visualization - use original scale if transformation was applied
            if transform:
                data_source = trainer.original_data
            else:
                data_source = trainer.data

            split_point = int(len(data_source) * (1 - validation_split))
            train_data_plot = data_source.iloc[:split_point]
            test_data_plot = data_source.iloc[split_point:]

            # Plot training data
            plt.plot(
                train_data_plot.index,
                train_data_plot.values,
                label="Training Data",
                alpha=0.7,
                color="blue",
            )

            # Plot ARIMA fit (only for training period) - inverse-transform if needed
            model = result["model"]
            if transform:
                # Inverse transform fittedvalues to original scale
                if callable(model.fittedvalues):
                    fitted_vals = model.fittedvalues()
                else:
                    fitted_vals = model.fittedvalues
                # Trim to match train_data_plot length
                fitted_vals = fitted_vals[: len(train_data_plot)]
                model_fitted = trainer._inverse_transform(fitted_vals)
            else:
                if callable(model.fittedvalues):
                    model_fitted = model.fittedvalues()
                else:
                    model_fitted = model.fittedvalues
                # Trim to match train_data_plot length
                model_fitted = model_fitted[: len(train_data_plot)]

            plt.plot(
                train_data_plot.index,
                model_fitted,
                label="ARIMA Fit",
                alpha=0.9,
                linewidth=2,
                color="orange",
            )

            # Plot test data
            plt.plot(
                test_data_plot.index,
                test_data_plot.values,
                label="Test Data",
                alpha=0.7,
                color="green",
                linestyle="--",
            )

            plt.title(
                f"ARIMA Model Training - {ticker.upper()} (p={result['parameters']['p']},d={result['parameters']['d']},q={result['parameters']['q']})"
            )
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            img_bytes = buf.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode()
            training_plot = ImageContent(
                type="image", data=img_base64, mimeType="image/png"
            )
            buf.close()
            plt.close()

            # Format comprehensive analysis result
            performance = result["performance"]
            model_info = result["parameters"]

            result_text = f"""🤖 ARIMA MODEL TRAINING - {ticker.upper()}

📊 TRAINING SUMMARY:
• Ticker: {ticker}
• Training Data Points: {performance["train_size"]}
• Validation Data Points: {performance["test_size"]}
• Training Split: {(1 - performance["validation_split"]) * 100:.0f}% train / {performance["validation_split"] * 100:.0f}% test
• Model Orders: ARIMA({model_info["p"]},{model_info["d"]},{model_info["q"]})

📈 MODEL PERFORMANCE METRICS:
• AIC: {result["aic"]:.4f}
• BIC: {result["bic"]:.4f}
• Log-Likelihood: {result["log_likelihood"]:.2f}
• Model Converged: {"✅ Yes" if result["converged"] else "❌ No"}

📊 VALIDATION METRICS (Test Set):
• Mean Squared Error (MSE): {performance["mse"]:.6f}
• Mean Absolute Error (MAE): {performance["mae"]:.4f}
• Mean Absolute Percentage Error (MAPE): {performance["mape"]:.2f}%

💡 MODEL INSIGHTS:
• Parameter Selection: {"Automatic" if auto_select else "Manual"}
• Next Steps: Use 'forecast_arima_model' for predictions, 'arima_model_diagnostics' for validation
• Alternative: Try different (p,d,q) combinations if performance unsatisfactory

📈 TRAINING VISUALIZATION:
• Historical data with fitted ARIMA model overlay
• Model fit diagnostics displayed
• Professional chart with training/validation split visualization
            """.strip()

            return [result_text.strip(), training_plot]

        except Exception as e:
            # Create error visualization
            error_image = PILImage.new("RGB", (600, 200), color="red")
            error_content = _encode_image(error_image)

            return [f"❌ ARIMA training failed for {ticker}: {str(e)}", error_content]

    except Exception as e:
        # Create error visualization
        error_image = PILImage.new("RGB", (600, 200), color="red")
        error_content = _encode_image(error_image)

        return [f"Error training ARIMA model for {ticker}: {str(e)}", error_content]


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


@mcp.tool()
async def arima_model_diagnostics(
    ticker: str, period: str = "1y", transform: str = None
) -> list:
    """
    Perform comprehensive diagnostics on trained ARIMA model.

    Provides:
    - Residual analysis with ACF/PACF plots
    - Normality tests and QQ plots
    - Ljung-Box test for autocorrelation
    - Model adequacy checks and recommendations
    - Visual diagnostic charts

    Args:
        ticker: Stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
        period: Time period for analysis ('1mo', '3mo', '6mo', '1y', '2y', '5y')
        transform: Data transformation type ("log", "boxcox", or None for no transformation)

    Returns:
        List containing text analysis and ImageContent with diagnostic plots
    """
    try:
        from src.model_training import ARIMATrainer

        # Initialize trainer
        trainer = ARIMATrainer(ticker, period)

        # Train model first before diagnostics
        try:
            trainer.train_model(
                p=None,
                d=1,
                q=None,
                validation_split=0.8,
                auto_select=True,
                transform=transform,
            )
        except Exception as e:
            # Create error visualization
            error_image = PILImage.new("RGB", (600, 200), color="red")
            error_content = _encode_image(error_image)
            return [
                f"❌ Failed to train ARIMA model for {ticker}: {str(e)}",
                error_content,
            ]

        # Perform diagnostics
        try:
            diagnostic_result = trainer.comprehensive_diagnostics()

            # Create diagnostic visualization
            buf = io.BytesIO()
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"ARIMA Model Diagnostics - {ticker.upper()}", fontsize=16)

            # Residuals Time Series
            axes[0, 0].plot(diagnostic_result["dates"], diagnostic_result["residuals"])
            axes[0, 0].axhline(y=0, color="r", linestyle="--")
            axes[0, 0].set_title("Residuals Over Time")
            axes[0, 0].set_xlabel("Date")
            axes[0, 0].set_ylabel("Residuals")
            axes[0, 0].grid(True, alpha=0.3)

            # Residual ACF
            axes[0, 1].stem(
                diagnostic_result["residual_acf"]["lags"][:20],
                diagnostic_result["residual_acf"]["acf_values"][:20],
                basefmt=" ",
            )
            axes[0, 1].axhline(y=0, color="black", linewidth=0.5)
            axes[0, 1].axhline(
                y=diagnostic_result["residual_acf"]["confidence_interval"],
                color="red",
                linestyle="--",
                alpha=0.5,
            )
            axes[0, 1].axhline(
                y=-diagnostic_result["residual_acf"]["confidence_interval"],
                color="red",
                linestyle="--",
                alpha=0.5,
            )
            axes[0, 1].set_title("Residual Autocorrelation")
            axes[0, 1].set_xlabel("Lag")
            axes[0, 1].set_ylabel("ACF")
            axes[0, 1].grid(True, alpha=0.3)

            # QQ Plot
            import scipy.stats as stats

            stats.probplot(diagnostic_result["residuals"], dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title("Normality QQ Plot")
            axes[1, 0].grid(True, alpha=0.3)

            # Residual Histogram
            axes[1, 1].hist(
                diagnostic_result["residuals"], bins=30, alpha=0.7, density=True
            )
            axes[1, 1].set_title("Residual Distribution")
            axes[1, 1].set_xlabel("Residuals")
            axes[1, 1].set_ylabel("Density")
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            img_bytes = buf.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode()
            diagnostic_plot = ImageContent(
                type="image", data=img_base64, mimeType="image/png"
            )
            buf.close()
            plt.close()

            # Format comprehensive diagnostic result
            diagnostics = diagnostic_result["diagnostics"]
            normality = diagnostic_result["normality_tests"]
            ljung_box = diagnostic_result["ljung_box_test"]
            recommendations = diagnostic_result["recommendations"]

            result_text = f"""🔍 ARIMA MODEL DIAGNOSTICS - {ticker.upper()}

📊 MODEL OVERVIEW:
• Ticker: {ticker}
• Data Points: {diagnostic_result["data_points"]}
• Model Orders: ARIMA({diagnostic_result["model_info"]["p"]},{diagnostic_result["model_info"]["d"]},{diagnostic_result["model_info"]["q"]})
• Analysis Period: {diagnostic_result["date_range"]["start"]} to {diagnostic_result["date_range"]["end"]}

📈 RESIDUAL ANALYSIS:
• Mean Residual: {diagnostics["mean_residual"]:.6f}
• Std Deviation: {diagnostics["std_residual"]:.6f}
• Min Residual: {diagnostics["min_residual"]:.6f}
• Max Residual: {diagnostics["max_residual"]:.6f}
• Residual Sum: {diagnostics["residual_sum"]:.6f}

🔬 NORMALITY TESTS:"""

            for test_name, test_result in normality.items():
                result_text += f"\n• {test_name}:"
                result_text += f"\n  - Statistic: {test_result['statistic']:.4f}"
                result_text += f"\n  - P-Value: {test_result['p_value']:.4f}"
                result_text += f"\n  - Result: {test_result['result']}"

            result_text += f"""

📊 LJUNG-BOX TEST:
• Test Statistic: {ljung_box["statistic"]:.4f}
• P-Value: {ljung_box["p_value"]:.4f}
• Lags Used: {ljung_box["lags"]}
• Result: {ljung_box["result"]}

✅ MODEL ADEQUACY:
• Overall Assessment: {recommendations["overall_assessment"]}
• White Noise: {recommendations["white_noise_conclusion"]}
• Autocorrelation: {recommendations["autocorrelation_conclusion"]}
• Normality: {recommendations["normality_conclusion"]}"""

            if recommendations["significant_residual_lags"]:
                result_text += f"\n• Significant Residual Lags: {recommendations['significant_residual_lags']}"

            result_text += """

💡 MODEL RECOMMENDATIONS:"""

            if recommendations["model_improvements"]:
                result_text += "\n🔧 Suggested Improvements:"
                for improvement in recommendations["model_improvements"]:
                    result_text += f"\n  • {improvement}"
            else:
                result_text += "\n✅ No major improvements needed"

            if recommendations["parameter_suggestions"]:
                result_text += "\n⚙️ Parameter Suggestions:"
                for suggestion in recommendations["parameter_suggestions"]:
                    result_text += f"\n  • {suggestion}"

            result_text += f"""

📋 QUALITY INDICATORS:
• Model Fit Quality: {recommendations["model_quality"]}
• Forecast Reliability: {recommendations["forecast_reliability"]}
• Complexity Level: {recommendations["complexity_level"]}
• Risk Assessment: {recommendations["risk_assessment"]}

⚠️ LIMITATIONS & WARNINGS:"""

            if recommendations["warnings"]:
                for warning in recommendations["warnings"]:
                    result_text += f"\n• {warning}"
            else:
                result_text += "\n• No major concerns identified"

            result_text += """

🔍 DIAGNOSTIC VISUALIZATION:
• Residuals time series plot for pattern detection
• Autocorrelation function for independence check
• QQ plot for normality assessment
• Histogram for distribution analysis
• Professional statistical diagnostic suite

💡 NEXT STEPS:
• Use 'forecast_arima_model' for predictions if diagnostics are favorable
• Consider retraining with different parameters if issues detected
• Monitor forecast accuracy and model performance over time
• Complement with fundamental analysis for investment decisions
            """.strip()

            return [result_text.strip(), diagnostic_plot]

        except Exception as e:
            # Create error visualization
            error_image = PILImage.new("RGB", (600, 200), color="red")
            error_content = _encode_image(error_image)

            return [
                f"❌ ARIMA diagnostics failed for {ticker}: {str(e)}",
                error_content,
            ]

    except Exception as e:
        # Create error visualization
        error_image = PILImage.new("RGB", (600, 200), color="red")
        error_content = _encode_image(error_image)

        return [
            f"Error generating ARIMA diagnostics for {ticker}: {str(e)}",
            error_content,
        ]


@mcp.tool()
async def forecast_arima_model(
    ticker: str,
    periods: int = 20,
    confidence: float = 0.95,
    p: int = 1,
    d: int = 1,
    q: int = 1,
    auto_select: bool = True,
    lags: int = 40,
    period: str = "1y",
    transform: str = None,
) -> list:
    """
    Generate ARIMA model forecasts with confidence intervals and validation.

    Provides:
    - Multi-period forecasting with confidence bands
    - Model validation and quality checks
    - Visual forecast charts with historical data
    - Performance metrics and accuracy indicators
    - Error handling with fallback strategies

    Args:
        ticker: Stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
        periods: Number of periods to forecast (default: 20 trading days)
        confidence: Confidence interval level (0.8-0.99, default: 0.95)
        p: AR order (None for auto-selection)
        d: Differencing order (default 1 for stock prices)
        q: MA order (None for auto-selection)
        auto_select: Use pmdarima auto_arima for parameter selection (True) or manual (False)
        lags: Number of lags for analysis (default 40, used for fallback ACF/PACF)
        period: Time period for training data ('1mo', '3mo', '6mo', '1y', '2y', '5y')
        transform: Data transformation type ("log", "boxcox", or None for no transformation)

    Returns:
        List containing text analysis and ImageContent with forecast plot
    """
    try:
        from src.model_training import ARIMATrainer

        # Validate inputs
        if periods < 1 or periods > 252:  # Max one year of trading days
            raise ValueError(f"Periods must be between 1 and 252, got {periods}")
        if not 0.8 <= confidence <= 0.99:
            raise ValueError(
                f"Confidence must be between 0.8 and 0.99, got {confidence}"
            )

        # Initialize trainer
        trainer = ARIMATrainer(ticker, period)

        # Check if model exists in cache
        model_key = f"{ticker}_{period}_{p}_{d}_{q}_{transform or 'none'}"
        cached_model = trainer.get_cached_model(model_key)

        if not cached_model:
            # Train model with auto_select and transform parameters
            train_result = trainer.train_model(
                p=p if not auto_select else None,
                d=d,
                q=q if not auto_select else None,
                validation_split=0.8,
                auto_select=auto_select,
                transform=transform,
            )
            model = train_result["model"]
            # Update model_key with actual trained parameters
            actual_params = train_result["parameters"]
            model_key = f"{ticker}_{period}_{actual_params['p']}_{actual_params['d']}_{actual_params['q']}_{transform or 'none'}"
        else:
            model = cached_model["model"]

        try:
            # Generate forecasts
            forecast_result = trainer.forecast_model(model, periods, confidence)

            # Create forecast visualization
            buf = io.BytesIO()
            plt.figure(figsize=(14, 8))

            # Historical data - use original scale if transformation was applied
            if transform:
                historical_data = trainer.original_data
            else:
                historical_data = trainer.data

            plt.plot(
                historical_data.index[-60:],
                historical_data.values[-60:],
                label="Historical Data",
                alpha=0.7,
                linewidth=2,
            )

            # Forecast
            forecast_dates = forecast_result["forecast_dates"]
            forecast_mean = forecast_result["forecast_mean"]
            forecast_ci_lower = forecast_result["forecast_ci_lower"]
            forecast_ci_upper = forecast_result["forecast_ci_upper"]

            # Convert Series to list for plotting compatibility
            forecast_mean_list = (
                forecast_mean.tolist()
                if hasattr(forecast_mean, "tolist")
                else list(forecast_mean)
            )

            plt.plot(
                forecast_dates,
                forecast_mean_list,
                label="Forecast",
                color="red",
                linewidth=2,
                marker="o",
            )
            plt.fill_between(
                forecast_dates,
                forecast_ci_lower,
                forecast_ci_upper,
                alpha=0.3,
                color="red",
                label=f"{int(confidence * 100)}% Confidence Band",
            )

            # Last known price line - use original scale if transformation was applied
            last_price = float(trainer.original_data.iloc[-1])
            plt.axhline(
                y=last_price,
                color="green",
                linestyle="--",
                alpha=0.7,
                label=f"Last Price: ₹{last_price:.2f}",
            )

            plt.title(f"ARIMA Forecast - {ticker.upper()} ({periods} periods)")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            img_bytes = buf.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode()
            forecast_plot = ImageContent(
                type="image", data=img_base64, mimeType="image/png"
            )
            buf.close()
            plt.close()

            # Format comprehensive analysis result
            forecast_analysis = forecast_result["analysis"]
            performance = forecast_result["performance"]

            result_text = f"""🔮 ARIMA FORECAST - {ticker.upper()}

📊 FORECAST SUMMARY:
• Ticker: {ticker}
• Forecast Periods: {periods} trading days
• Confidence Level: {confidence * 100:.0f}%
• Last Price: ₹{forecast_analysis["last_price"]:.2f}
• Forecast Horizon: {forecast_analysis["forecast_start_date"]} to {forecast_analysis["forecast_end_date"]}

📈 FORECAST RESULTS:
• Final Forecast: ₹{forecast_analysis["final_forecast"]:.2f}
• Price Change: {forecast_analysis["price_change"]:+.2f} ({forecast_analysis["price_change_percent"]:+.2f}%)
• Min Forecast: ₹{forecast_analysis["min_forecast"]:.2f}
• Max Forecast: ₹{forecast_analysis["max_forecast"]:.2f}
• Forecast Range: ₹{forecast_analysis["forecast_range"]:.2f}

📊 CONFIDENCE INTERVALS:
• Lower Bound: ₹{forecast_analysis["ci_lower_bound"]:.2f}
• Upper Bound: ₹{forecast_analysis["ci_upper_bound"]:.2f}
• Band Width: ₹{forecast_analysis["ci_band_width"]:.2f}
• Relative Band Width: {forecast_analysis["relative_band_width"]:.2f}%

🎯 FORECAST ACCURACY INDICATORS:
• Standard Error: {performance["standard_error"]:.4f}
• Mean Absolute Error: {performance["mae"]:.4f}
• Prediction Quality: {forecast_analysis["prediction_quality"]}

💡 TRADING IMPLICATIONS:"""

            if forecast_analysis["price_change_percent"] > 5:
                result_text += f"\n🟢 BULLISH FORECAST: Expected {forecast_analysis['price_change_percent']:+.2f}% movement"
            elif forecast_analysis["price_change_percent"] < -5:
                result_text += f"\n🔴 BEARISH FORECAST: Expected {forecast_analysis['price_change_percent']:+.2f}% movement"
            else:
                result_text += f"\n🟡 NEUTRAL FORECAST: Expected {forecast_analysis['price_change_percent']:+.2f}% movement"

            if forecast_analysis["relative_band_width"] > 0.15:
                result_text += "\n⚠️ HIGH UNCERTAINTY: Wide confidence bands indicate forecast uncertainty"
            elif forecast_analysis["relative_band_width"] < 0.05:
                result_text += "\n✅ HIGH CONFIDENCE: Narrow confidence bands suggest reliable forecast"
            else:
                result_text += (
                    "\n📊 MODERATE CONFIDENCE: Reasonable forecast uncertainty"
                )

            result_text += f"""

 📋 MODEL PERFORMANCE:
 • Training Data Points: {performance["data_points"]}
  • Model Convergence: {"✅ Converged" if trainer._get_model_convergence(model) else "❌ Non-converged"}
 • Model Quality: {forecast_analysis["model_quality"]}

🔍 RISK CONSIDERATIONS:"""

            if forecast_analysis["price_volatility"] > 0.25:
                result_text += f"\n• High Volatility (σ={forecast_analysis['price_volatility']:.1%}) - Higher risk expected"
            elif forecast_analysis["price_volatility"] > 0.15:
                result_text += f"\n• Moderate Volatility (σ={forecast_analysis['price_volatility']:.1%}) - Normal market conditions"
            else:
                result_text += f"\n• Low Volatility (σ={forecast_analysis['price_volatility']:.1%}) - Stable conditions"

            result_text += f"""
• Forecast Validity: Next {periods} trading days only
• Market Conditions: Forecast assumes normal market conditions
• External Events: Not accounted for in statistical forecast

⚙️ RECOMMENDATIONS:
• Use forecast as one input among multiple analysis methods
• Monitor actual price movements vs forecast for validation
• Consider fundamental analysis and market sentiment
• Set appropriate stop-loss levels based on forecast uncertainty
• Re-run forecast with new data periodically

📈 FORECAST VISUALIZATION:
• Historical price data with ARIMA model forecast
• Confidence bands showing prediction uncertainty
• Professional time series forecasting chart
            """.strip()

            return [result_text.strip(), forecast_plot]

        except Exception as e:
            # Create error visualization
            error_image = PILImage.new("RGB", (600, 200), color="red")
            error_content = _encode_image(error_image)

            return [
                f"❌ ARIMA forecasting failed for {ticker}: {str(e)}",
                error_content,
            ]

    except Exception as e:
        # Create error visualization
        error_image = PILImage.new("RGB", (600, 200), color="red")
        error_content = _encode_image(error_image)

        return [
            f"Error generating ARIMA forecast for {ticker}: {str(e)}",
            error_content,
        ]


@mcp.tool()
async def forecast_prophet_model(
    ticker: str,
    periods: int = 20,
    confidence: float = 0.95,
    period: str = "1y",
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    seasonality_mode: str = "additive",
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    holidays_prior_scale: float = 10.0,
    validation_split: float = 0.2,
    include_holidays: bool = False,
) -> list:
    """
    Train Prophet model and generate forecasts with confidence intervals.

    Provides:
    - Automatic seasonality detection (yearly, weekly patterns)
    - Trend changepoint identification
    - Holiday effects support (Indian market holidays)
    - Component decomposition (trend + seasonality)
    - Multi-period forecasting with confidence bands
    - Model validation and quality checks
    - Visual forecast charts with historical data

    Args:
        ticker: Stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
        periods: Number of periods to forecast (default: 20 trading days)
        confidence: Confidence interval level (0.8-0.99, default: 0.95)
        period: Time period for training data ('1mo', '3mo', '6mo', '1y', '2y', '5y')
        yearly_seasonality: Enable yearly seasonality (default True)
        weekly_seasonality: Enable weekly seasonality (default True)
        seasonality_mode: 'additive' or 'multiplicative' (default 'additive')
        changepoint_prior_scale: Flexibility of trend changes (default 0.05)
        seasonality_prior_scale: Flexibility of seasonality (default 10.0)
        holidays_prior_scale: Flexibility of holiday effects (default 10.0)
        validation_split: Train-validation split ratio (default 0.2)
        include_holidays: Include Indian market holidays (default False)

    Returns:
        List containing text analysis and ImageContent with forecast plot
    """
    try:
        from src.model_training import ProphetTrainer

        if periods < 1 or periods > 252:
            raise ValueError(f"Periods must be between 1 and 252, got {periods}")
        if not 0.8 <= confidence <= 0.99:
            raise ValueError(
                f"Confidence must be between 0.8 and 0.99, got {confidence}"
            )

        trainer = ProphetTrainer(ticker, period)

        train_result = trainer.train_model(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=False,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            holidays=None,
            validation_split=validation_split,
            include_holidays=include_holidays,
            confidence=confidence,
        )

        model = train_result["model"]
        train_forecast = train_result["train_forecast"]

        forecast_result = trainer.forecast(periods, model=model)

        try:
            buf = io.BytesIO()
            plt.figure(figsize=(14, 8))

            split_point = int(len(trainer.original_data) * (1 - validation_split))
            train_data_plot = trainer.original_data.iloc[:split_point]
            test_data_plot = trainer.original_data.iloc[split_point:]

            # Plot training data
            plt.plot(
                train_data_plot.index,
                train_data_plot.values,
                label="Training Data",
                alpha=0.7,
                linewidth=2,
                color="blue",
            )

            # Plot Prophet fit (aligned by date in train_forecast)
            if len(train_forecast) > 0:
                plt.plot(
                    train_forecast["ds"],
                    train_forecast["yhat"],
                    label="Prophet Fit (Training)",
                    alpha=0.9,
                    linewidth=2,
                    color="orange",
                )

            # Plot test data
            plt.plot(
                test_data_plot.index,
                test_data_plot.values,
                label="Test Data",
                alpha=0.7,
                color="green",
                linestyle="--",
            )

            forecast_dates = forecast_result["forecast_dates"]
            forecast_mean = forecast_result["forecast_mean"]
            forecast_ci_lower = forecast_result["forecast_ci_lower"]
            forecast_ci_upper = forecast_result["forecast_ci_upper"]

            # Convert DatetimeIndex to NumPy array for matplotlib compatibility
            forecast_dates_array = forecast_dates.to_numpy()

            forecast_mean_list = (
                forecast_mean.tolist()
                if hasattr(forecast_mean, "tolist")
                else list(forecast_mean)
            )

            plt.plot(
                forecast_dates_array,
                forecast_mean_list,
                label="Forecast",
                color="red",
                linewidth=2,
                marker="o",
            )
            plt.fill_between(
                forecast_dates_array,
                forecast_ci_lower,
                forecast_ci_upper,
                alpha=0.3,
                color="red",
                label=f"{int(confidence * 100)}% Confidence Band",
            )

            last_price = float(trainer.original_data.iloc[-1])
            plt.axhline(
                y=last_price,
                color="green",
                linestyle="--",
                alpha=0.7,
                label=f"Last Price: ₹{last_price:.2f}",
            )

            plt.title(f"Prophet Forecast - {ticker.upper()} ({periods} trading days)")
            plt.xlabel("Date")
            plt.ylabel("Price (₹)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            img_bytes = buf.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode()
            forecast_plot = ImageContent(
                type="image", data=img_base64, mimeType="image/png"
            )
            buf.close()
            plt.close()

            forecast_analysis = forecast_result["analysis"]
            performance = forecast_result["performance"]
            parameters = train_result["parameters"]

            result_text = f"""🔮 PROPHET FORECAST - {ticker.upper()}

📊 MODEL PARAMETERS:
• Yearly Seasonality: {"Enabled" if parameters["yearly_seasonality"] else "Disabled"}
• Weekly Seasonality: {"Enabled" if parameters["weekly_seasonality"] else "Disabled"}
• Seasonality Mode: {parameters["seasonality_mode"].title()}
• Changepoint Prior Scale: {parameters["changepoint_prior_scale"]}
• Seasonality Prior Scale: {parameters["seasonality_prior_scale"]}
• Indian Holidays: {"Included" if parameters["include_holidays"] else "Excluded"}

📊 TRAINING SUMMARY:
• Training Data Points: {train_result["train_data_points"]}
• Validation Data Points: {train_result["test_data_points"]}
• Training Split: {(1 - validation_split) * 100:.0f}% train / {validation_split * 100:.0f}% test

📈 VALIDATION METRICS:
• Mean Squared Error (MSE): {train_result["performance"]["mse"]:.6f}
• Mean Absolute Error (MAE): {train_result["performance"]["mae"]:.4f}
• Mean Absolute Percentage Error (MAPE): {train_result["performance"]["mape"]:.2f}%

📊 FORECAST SUMMARY:
• Ticker: {ticker}
• Forecast Periods: {periods} trading days
• Confidence Level: {confidence * 100:.0f}%
• Last Price: ₹{forecast_analysis["last_price"]:.2f}
• Forecast Horizon: {forecast_analysis["forecast_start_date"]} to {forecast_analysis["forecast_end_date"]}

📈 FORECAST RESULTS:
• Final Forecast: ₹{forecast_analysis["final_forecast"]:.2f}
• Price Change: {forecast_analysis["price_change"]:+.2f} ({forecast_analysis["price_change_percent"]:+.2f}%)
• Min Forecast: ₹{forecast_analysis["min_forecast"]:.2f}
• Max Forecast: ₹{forecast_analysis["max_forecast"]:.2f}
• Forecast Range: ₹{forecast_analysis["forecast_range"]:.2f}

📊 CONFIDENCE INTERVALS:
• Lower Bound: ₹{forecast_analysis["ci_lower_bound"]:.2f}
• Upper Bound: ₹{forecast_analysis["ci_upper_bound"]:.2f}
• Band Width: ₹{forecast_analysis["ci_band_width"]:.2f}
• Relative Band Width: {forecast_analysis["relative_band_width"]:.2f}%

🎯 FORECAST ACCURACY INDICATORS:
• Standard Error: {performance["standard_error"]:.4f}
• Mean Absolute Error: {performance["mae"]:.4f}
• Prediction Quality: {forecast_analysis["prediction_quality"]}

💡 TRADING IMPLICATIONS:"""

            if forecast_analysis["price_change_percent"] > 5:
                result_text += f"\n🟢 BULLISH FORECAST: Expected {forecast_analysis['price_change_percent']:+.2f}% movement"
            elif forecast_analysis["price_change_percent"] < -5:
                result_text += f"\n🔴 BEARISH FORECAST: Expected {forecast_analysis['price_change_percent']:+.2f}% movement"
            else:
                result_text += f"\n🟡 NEUTRAL FORECAST: Expected {forecast_analysis['price_change_percent']:+.2f}% movement"

            if forecast_analysis["relative_band_width"] > 0.15:
                result_text += "\n⚠️ HIGH UNCERTAINTY: Wide confidence bands indicate forecast uncertainty"
            elif forecast_analysis["relative_band_width"] < 0.05:
                result_text += "\n✅ HIGH CONFIDENCE: Narrow confidence bands suggest reliable forecast"
            else:
                result_text += (
                    "\n📊 MODERATE CONFIDENCE: Reasonable forecast uncertainty"
                )

            result_text += f"""

📋 MODEL PERFORMANCE:
• Training Data Points: {performance["data_points"]}
• Model Quality: {forecast_analysis["model_quality"]}

🔍 RISK CONSIDERATIONS:"""

            if forecast_analysis["price_volatility"] > 0.25:
                result_text += f"\n• High Volatility (σ={forecast_analysis['price_volatility']:.1%}) - Higher risk expected"
            elif forecast_analysis["price_volatility"] > 0.15:
                result_text += f"\n• Moderate Volatility (σ={forecast_analysis['price_volatility']:.1%}) - Normal market conditions"
            else:
                result_text += f"\n• Low Volatility (σ={forecast_analysis['price_volatility']:.1%}) - Stable conditions"

            result_text += f"""
• Forecast Validity: Next {periods} trading days only
• Market Conditions: Forecast assumes normal market conditions
• External Events: Not accounted for in statistical forecast

⚙️ RECOMMENDATIONS:
• Use forecast as one input among multiple analysis methods
• Monitor actual price movements vs forecast for validation
• Consider fundamental analysis and market sentiment
• Set appropriate stop-loss levels based on forecast uncertainty
• Re-run forecast with new data periodically

🤖 PROPHET ADVANTAGES:
• Automatic seasonality detection (no manual parameter tuning)
• Changepoint detection identifies trend changes
• Handles missing data gracefully
• Component decomposition (trend + seasonality)
• Holiday effects support for market events
• More interpretable than ARIMA/SARIMA

📈 FORECAST VISUALIZATION:
• Historical training data (blue)
• Prophet in-sample fit (orange)
• Test validation data (green dashed)
• Prophet forecast (red with dots)
• Confidence bands showing prediction uncertainty
• Last price reference line (green dashed)
• Professional time series forecasting chart
            """.strip()

            return [result_text.strip(), forecast_plot]

        except Exception as e:
            error_image = PILImage.new("RGB", (600, 200), color="red")
            error_content = _encode_image(error_image)

            return [
                f"❌ Prophet forecasting failed for {ticker}: {str(e)}",
                error_content,
            ]

    except Exception as e:
        error_image = PILImage.new("RGB", (600, 200), color="red")
        error_content = _encode_image(error_image)

        return [
            f"Error generating Prophet forecast for {ticker}: {str(e)}",
            error_content,
        ]


def main():
    """Initialize and run the server"""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
