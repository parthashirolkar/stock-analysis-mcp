"""
Indian Stock Market Data Analyzer

Core functionality for fetching Indian stock data from BSE/NSE.
Optimized for MCP server usage with FastMCP.
"""

import yfinance as yf
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image as PILImage
import io
from datetime import datetime
from typing import Dict, Any, List, Tuple


def normalize_ticker(ticker: str) -> Dict[str, str]:
    """
    Normalize Indian stock ticker for different APIs

    Args:
        ticker: Stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY.NS')

    Returns:
        Dictionary with normalized tickers for different sources
    """
    ticker = ticker.upper().strip()

    # Remove common suffixes and normalize
    ticker = re.sub(r"\.(BO|NS|BSE|NSE)$", "", ticker)

    return {
        "yfinance_nse": f"{ticker}.NS",
        "yfinance_bse": f"{ticker}.BO",
        "base": ticker,
    }


async def get_stock_quote(ticker: str) -> Dict[str, Any]:
    """
    Get current stock price and basic trading information

    Args:
        ticker: Indian stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')

    Returns:
        Dictionary with current stock price and trading data
    """
    ticker_variants = normalize_ticker(ticker)

    # Try NSE first, then BSE
    stock_data = None
    exchange_used = None

    for exchange, ticker_symbol in [
        ("NSE", ticker_variants["yfinance_nse"]),
        ("BSE", ticker_variants["yfinance_bse"]),
    ]:
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            if info and "regularMarketPrice" in info:
                stock_data = stock
                exchange_used = exchange
                break
        except Exception:
            continue

    if not stock_data:
        raise ValueError(f"Could not fetch data for ticker {ticker_variants['base']}")

    info = stock_data.info

    return {
        "ticker": ticker_variants["base"],
        "exchange": exchange_used,
        "current_price": info.get("regularMarketPrice", info.get("currentPrice", 0)),
        "change": info.get("regularMarketChangePercent", 0),
        "change_amount": info.get("regularMarketChange", 0),
        "high": info.get("dayHigh", info.get("regularMarketDayHigh", 0)),
        "low": info.get("dayLow", info.get("regularMarketDayLow", 0)),
        "open": info.get("regularMarketOpen", 0),
        "previous_close": info.get("regularMarketPreviousClose", 0),
        "volume": info.get("regularMarketVolume", info.get("volume", 0)),
        "market_cap": info.get("marketCap", 0),
        "currency": "INR",
        "last_updated": datetime.now().isoformat(),
    }


async def get_company_fundamentals(ticker: str) -> Dict[str, Any]:
    """
    Get fundamental analysis data for a company

    Args:
        ticker: Indian stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')

    Returns:
        Dictionary with fundamental metrics
    """
    ticker_variants = normalize_ticker(ticker)

    # Try NSE first, then BSE
    stock_data = None
    exchange_used = None

    for exchange, ticker_symbol in [
        ("NSE", ticker_variants["yfinance_nse"]),
        ("BSE", ticker_variants["yfinance_bse"]),
    ]:
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            if info and "regularMarketPrice" in info:
                stock_data = stock
                exchange_used = exchange
                break
        except Exception:
            continue

    if not stock_data:
        raise ValueError(f"Could not fetch data for ticker {ticker_variants['base']}")

    info = stock_data.info
    hist = stock_data.history(period="1y")

    # Calculate 52-week high/low from historical data
    fifty_two_week_high = hist["High"].max() if not hist.empty else 0
    fifty_two_week_low = hist["Low"].min() if not hist.empty else 0

    return {
        "ticker": ticker_variants["base"],
        "exchange": exchange_used,
        "company_name": info.get(
            "longName", info.get("shortName", ticker_variants["base"])
        ),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap", 0),
        "pe_ratio": info.get("trailingPE", info.get("forwardPE", None)),
        "pb_ratio": info.get("priceToBook", None),
        "eps": info.get("trailingEps", info.get("forwardEps", None)),
        "book_value": info.get("bookValue", None),
        "roe": info.get("returnOnEquity", None),
        "dividend_yield": info.get("dividendYield", None),
        "beta": info.get("beta", None),
        "debt_to_equity": info.get("debtToEquity", None),
        "current_ratio": info.get("currentRatio", None),
        "profit_margin": info.get("profitMargins", None),
        "operating_margin": info.get("operatingMargins", None),
        "52_week_high": fifty_two_week_high,
        "52_week_low": fifty_two_week_low,
        "avg_volume": info.get("averageVolume", 0),
        "website": info.get("website", "N/A"),
        "business_summary": info.get("longBusinessSummary", "N/A"),
        "last_updated": datetime.now().isoformat(),
    }


async def get_stock_news(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get recent news articles for a specific stock

    Args:
        ticker: Indian stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
        limit: Maximum number of news articles to return

    Returns:
        List of news article dictionaries
    """
    ticker_variants = normalize_ticker(ticker)
    news_items = []

    # Try to get news from yfinance
    for exchange_suffix in [".NS", ".BO"]:
        try:
            stock = yf.Ticker(f"{ticker_variants['base']}{exchange_suffix}")
            news = stock.news
            if news:
                for item in news[:limit]:
                    # Extract data from nested content structure
                    content = item.get("content", {})
                    provider = content.get("provider", {})

                    news_items.append(
                        {
                            "title": content.get("title", "No title available"),
                            "url": content.get("clickThroughUrl", {}).get("url", ""),
                            "publisher": provider.get("displayName", "Unknown"),
                            "published": content.get("pubDate", ""),
                            "summary": content.get("summary", "No summary available"),
                        }
                    )
                break
        except Exception:
            continue

    # Fallback to general Indian market news if no specific news found
    if not news_items:
        news_items = [
            {
                "title": f"General market information for {ticker_variants['base']}",
                "url": f"https://www.moneycontrol.com/india/stockpricequote/{ticker_variants['base']}",
                "publisher": "MoneyControl",
                "published": datetime.now().timestamp(),
                "summary": f"Visit MoneyControl for detailed information about {ticker_variants['base']}",
            }
        ]

    return news_items


async def search_stocks(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search for Indian stocks by company name or ticker symbol

    Args:
        query: Search query - company name or partial ticker symbol
        limit: Maximum number of results to return

    Returns:
        List of matching stocks with basic information
    """
    # This is a simplified implementation - in production, you'd use a proper search API
    # For now, return some popular Indian stocks that match the query

    popular_stocks = [
        {"ticker": "RELIANCE", "name": "Reliance Industries Ltd", "sector": "Energy"},
        {"ticker": "TCS", "name": "Tata Consultancy Services", "sector": "Technology"},
        {"ticker": "INFY", "name": "Infosys Ltd", "sector": "Technology"},
        {"ticker": "HDFC", "name": "HDFC Bank Ltd", "sector": "Banking"},
        {"ticker": "ICICI", "name": "ICICI Bank Ltd", "sector": "Banking"},
        {
            "ticker": "HINDUNILVR",
            "name": "Hindustan Unilever Ltd",
            "sector": "Consumer Goods",
        },
        {"ticker": "SBIN", "name": "State Bank of India", "sector": "Banking"},
        {"ticker": "BHARTIARTL", "name": "Bharti Airtel Ltd", "sector": "Telecom"},
        {"ticker": "ITC", "name": "ITC Ltd", "sector": "Consumer Goods"},
        {"ticker": "KOTAKBANK", "name": "Kotak Mahindra Bank", "sector": "Banking"},
    ]

    query = query.upper().strip()
    results = []

    for stock in popular_stocks:
        if (
            query in stock["ticker"]
            or query in stock["name"].upper()
            or query in stock["sector"].upper()
        ):
            results.append(stock)
            if len(results) >= limit:
                break

    return results


async def get_historical_data(ticker: str, period: str = "1M") -> Dict[str, Any]:
    """
    Get historical price data for technical analysis

    Args:
        ticker: Indian stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
        period: Time period for historical data (1D, 1W, 1M, 3M, 6M, 1Y, 2Y, 5Y)

    Returns:
        Dictionary with historical OHLC data
    """
    ticker_variants = normalize_ticker(ticker)

    # Map period strings to yfinance periods
    period_mapping = {
        "1D": "1d",
        "1W": "5d",
        "1M": "1mo",
        "3M": "3mo",
        "6M": "6mo",
        "1Y": "1y",
        "2Y": "2y",
        "5Y": "5y",
    }

    yf_period = period_mapping.get(period, "1mo")

    # Try NSE first, then BSE
    for ticker_symbol in [
        ticker_variants["yfinance_nse"],
        ticker_variants["yfinance_bse"],
    ]:
        try:
            stock = yf.Ticker(ticker_symbol)
            hist = stock.history(period=yf_period)

            if not hist.empty:
                # Convert to list of dictionaries
                data = []
                for date, row in hist.iterrows():
                    data.append(
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "open": round(float(row["Open"]), 2),
                            "high": round(float(row["High"]), 2),
                            "low": round(float(row["Low"]), 2),
                            "close": round(float(row["Close"]), 2),
                            "volume": int(row["Volume"]),
                        }
                    )

                return {
                    "ticker": ticker_variants["base"],
                    "period": period,
                    "data": data,
                    "last_updated": datetime.now().isoformat(),
                }
        except Exception:
            continue

    raise ValueError(
        f"Could not fetch historical data for ticker {ticker_variants['base']}"
    )


async def get_market_overview() -> Dict[str, Any]:
    """
    Get current Indian market indices and sector performance

    Returns:
        Dictionary with market indices data
    """
    # Major Indian indices
    indices = {
        "^NSEI": {"name": "NIFTY 50", "symbol": "NSEI"},
        "^BSESN": {"name": "SENSEX", "symbol": "BSESN"},
        "^NSEBANK": {"name": "NIFTY BANK", "symbol": "NSEBANK"},
        "^CNXIT": {"name": "NIFTY IT", "symbol": "CNXIT"},
    }

    market_data = {}

    for index_symbol, index_info in indices.items():
        try:
            index = yf.Ticker(index_symbol)
            info = index.info

            market_data[index_info["symbol"]] = {
                "name": index_info["name"],
                "current_value": info.get("regularMarketPrice", 0),
                "change": info.get("regularMarketChangePercent", 0),
                "change_amount": info.get("regularMarketChange", 0),
                "high": info.get("dayHigh", 0),
                "low": info.get("dayLow", 0),
                "volume": info.get("regularMarketVolume", 0),
            }
        except Exception as e:
            market_data[index_info["symbol"]] = {
                "name": index_info["name"],
                "error": str(e),
            }

    return {
        "indices": market_data,
        "market_status": await get_market_status(),
        "last_updated": datetime.now().isoformat(),
    }


async def get_market_status() -> Dict[str, Any]:
    """
    Get current Indian market status and trading hours

    Returns:
        Dictionary with market status information
    """
    now = datetime.now()

    # Indian market hours: 9:15 AM to 3:30 PM IST, Monday to Friday
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

    # Check if it's a weekday (Monday=0, Friday=4)
    is_weekday = now.weekday() < 5

    if is_weekday and market_open <= now <= market_close:
        status = "Open"
        status_description = "Market is currently open for trading"
    elif is_weekday and now < market_open:
        status = "Pre-Market"
        status_description = f"Market opens at {market_open.strftime('%I:%M %p')} IST"
    elif is_weekday and now > market_close:
        status = "Closed"
        status_description = (
            f"Market closed. Opens tomorrow at {market_open.strftime('%I:%M %p')} IST"
        )
    else:
        status = "Weekend"
        status_description = "Market is closed for the weekend"

    return {
        "status": status,
        "description": status_description,
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S IST"),
        "market_open": market_open.strftime("%I:%M %p"),
        "market_close": market_close.strftime("%I:%M %p"),
        "timezone": "IST (Indian Standard Time)",
    }


async def get_popular_stocks() -> List[Dict[str, Any]]:
    """
    Get list of frequently analyzed Indian stocks

    Returns:
        List of popular Indian stocks with basic information
    """
    return [
        {
            "ticker": "RELIANCE",
            "name": "Reliance Industries Ltd",
            "sector": "Energy",
            "description": "Largest Indian conglomerate",
        },
        {
            "ticker": "TCS",
            "name": "Tata Consultancy Services",
            "sector": "Technology",
            "description": "India's largest IT services company",
        },
        {
            "ticker": "INFY",
            "name": "Infosys Ltd",
            "sector": "Technology",
            "description": "Major IT services provider",
        },
        {
            "ticker": "HDFC",
            "name": "HDFC Bank Ltd",
            "sector": "Banking",
            "description": "India's largest private sector bank",
        },
        {
            "ticker": "ICICI",
            "name": "ICICI Bank Ltd",
            "sector": "Banking",
            "description": "Major private sector bank",
        },
        {
            "ticker": "HINDUNILVR",
            "name": "Hindustan Unilever Ltd",
            "sector": "Consumer Goods",
            "description": "Leading FMCG company",
        },
        {
            "ticker": "SBIN",
            "name": "State Bank of India",
            "sector": "Banking",
            "description": "India's largest public sector bank",
        },
        {
            "ticker": "BHARTIARTL",
            "name": "Bharti Airtel Ltd",
            "sector": "Telecom",
            "description": "Major telecommunications provider",
        },
        {
            "ticker": "ITC",
            "name": "ITC Ltd",
            "sector": "Consumer Goods",
            "description": "Diversified conglomerate",
        },
        {
            "ticker": "KOTAKBANK",
            "name": "Kotak Mahindra Bank",
            "sector": "Banking",
            "description": "Leading private sector bank",
        },
    ]


def calculate_bollinger_bands(
    data: pd.DataFrame, window: int = 20, std_dev: float = 2
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands for stock data

    Args:
        data: DataFrame with OHLC data
        window: Moving average window (default 20)
        std_dev: Standard deviation multiplier (default 2)

    Returns:
        DataFrame with Bollinger Bands added
    """
    df = data.copy()

    # Calculate SMA and Standard Deviation
    df["SMA"] = df["Close"].rolling(window=window).mean()
    df["SD"] = df["Close"].rolling(window=window).std()

    # Calculate Upper and Lower Bands
    df["Upper Band"] = df["SMA"] + (df["SD"] * std_dev)
    df["Lower Band"] = df["SMA"] - (df["SD"] * std_dev)

    # Calculate additional metrics
    df["Band_Width"] = df["Upper Band"] - df["Lower Band"]
    df["Band_Width_Percent"] = (df["Band_Width"] / df["Close"]) * 100
    df["Band_Position"] = (
        (df["Close"] - df["Lower Band"]) / (df["Upper Band"] - df["Lower Band"])
    ) * 100

    return df


def create_bollinger_chart(
    symbol: str, period: str = "3mo", interval: str = "1d"
) -> Tuple[PILImage.Image, Dict[str, Any]]:
    """
    Generate Bollinger Bands chart for Indian stock

    Args:
        symbol: Stock ticker symbol
        period: Time period (1mo, 3mo, 6mo, 1y, 2y)
        interval: Data interval (1d, 1h, 5m)

    Returns:
        Tuple of (PIL Image, analysis_summary)
    """
    try:
        # Set matplotlib style for better looking charts
        plt.style.use("default")
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"

        # Get ticker variants for Yahoo Finance
        ticker_variants = normalize_ticker(symbol)

        # Try NSE first, then BSE
        ticker_symbol = ticker_variants.get(
            "yfinance_nse", ticker_variants.get("yfinance_bse", f"{symbol}.NS")
        )

        # Fetch historical data
        stock = yf.Ticker(ticker_symbol)
        data = stock.history(period=period, interval=interval)

        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")

        # Calculate Bollinger Bands
        data = calculate_bollinger_bands(data)

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}
        )
        fig.suptitle(
            f"{symbol.upper()} - Bollinger Bands Analysis",
            fontsize=16,
            fontweight="bold",
        )

        # Plot 1: Price and Bollinger Bands
        # Plot the shaded area between bands first
        ax1.fill_between(
            data.index,
            data["Upper Band"],
            data["Lower Band"],
            alpha=0.1,
            color="gray",
            label="Band Range",
        )

        # Plot the bands
        ax1.plot(
            data.index,
            data["Upper Band"],
            "r--",
            label="Upper Band",
            alpha=0.8,
            linewidth=1.5,
        )
        ax1.plot(
            data.index,
            data["Lower Band"],
            "g--",
            label="Lower Band",
            alpha=0.8,
            linewidth=1.5,
        )
        ax1.plot(
            data.index, data["SMA"], "b-", label="20-day SMA", alpha=0.9, linewidth=2
        )

        # Plot the price
        ax1.plot(data.index, data["Close"], "k-", label="Close Price", linewidth=2.5)

        # Highlight current position
        current_price = data["Close"].iloc[-1]
        current_position = data["Band_Position"].iloc[-1]
        current_date = data.index[-1]

        # Color based on position
        if current_position > 75:
            position_color = "red"
            position_status = "Overbought"
        elif current_position < 25:
            position_color = "green"
            position_status = "Oversold"
        else:
            position_color = "blue"
            position_status = "Neutral"

        # Mark current price
        ax1.scatter(
            current_date,
            current_price,
            color=position_color,
            s=100,
            zorder=5,
            edgecolor="black",
            linewidth=2,
            label=f"Current ({position_status})",
        )

        # Add annotation for current position
        ax1.annotate(
            f"Position: {current_position:.1f}%\nRs.{current_price:.2f}",
            xy=(current_date, current_price),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=position_color, alpha=0.7),
            fontsize=9,
            fontweight="bold",
            color="white",
        )

        # Formatting
        ax1.set_ylabel("Price (Rs.)", fontsize=12)
        ax1.legend(loc="upper left", fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Price Action with Bollinger Bands", fontsize=14)

        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Plot 2: Band Width (Volatility)
        ax2.fill_between(
            data.index, data["Band_Width_Percent"], alpha=0.3, color="orange"
        )
        ax2.plot(
            data.index, data["Band_Width_Percent"], color="darkorange", linewidth=2
        )
        ax2.axhline(
            y=data["Band_Width_Percent"].mean(),
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Average ({data['Band_Width_Percent'].mean():.1f}%)",
        )

        # Highlight squeeze zones (low volatility)
        squeeze_threshold = data["Band_Width_Percent"].quantile(0.25)
        ax2.fill_between(
            data.index,
            0,
            squeeze_threshold,
            alpha=0.2,
            color="blue",
            label="Low Volatility Zone",
        )

        ax2.set_ylabel("Band Width (%)", fontsize=12)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_title("Band Width (Volatility Indicator)", fontsize=12)
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        # Adjust layout
        plt.tight_layout()

        # Convert matplotlib figure to PIL Image
        buffer = io.BytesIO()
        plt.savefig(
            buffer, format="png", dpi=150, bbox_inches="tight", facecolor="white"
        )
        buffer.seek(0)

        # Create PIL Image from buffer
        pil_image = PILImage.open(buffer)

        # Clear the plot to free memory
        plt.close(fig)

        # Generate analysis summary
        current_price = data["Close"].iloc[-1]
        current_band_width = data["Band_Width_Percent"].iloc[-1]
        avg_band_width = data["Band_Width_Percent"].mean()

        # Volatility analysis
        if current_band_width > avg_band_width * 1.2:
            volatility_status = "HIGH (expanding bands)"
        elif current_band_width < avg_band_width * 0.8:
            volatility_status = "LOW (narrowing bands) - Watch for breakout!"
        else:
            volatility_status = "NORMAL"

        # Check for band squeeze
        recent_avg_width = data["Band_Width_Percent"].tail(10).mean()
        squeeze_detected = recent_avg_width < squeeze_threshold

        analysis = {
            "symbol": symbol.upper(),
            "current_price": round(current_price, 2),
            "position_percentage": round(current_position, 1),
            "position_status": position_status,
            "current_band_width": round(current_band_width, 2),
            "average_band_width": round(avg_band_width, 2),
            "volatility_status": volatility_status,
            "squeeze_detected": squeeze_detected,
            "data_points": len(data),
            "period": period,
            "interval": interval,
            "date_range": {
                "start": data.index[0].strftime("%Y-%m-%d"),
                "end": data.index[-1].strftime("%Y-%m-%d"),
            },
            "price_range": {
                "min": round(data["Close"].min(), 2),
                "max": round(data["Close"].max(), 2),
            },
        }

        return pil_image, analysis

    except Exception as e:
        raise ValueError(f"Failed to generate Bollinger Bands chart: {str(e)}")


async def get_technical_indicators(ticker: str, period: str = "3mo") -> Dict[str, Any]:
    """
    Get technical analysis indicators for a stock

    Args:
        ticker: Indian stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
        period: Time period for analysis (1mo, 3mo, 6mo, 1y, 2y)

    Returns:
        Dictionary with technical indicators (RSI, MACD, SMA, EMA)
    """
    ticker_variants = normalize_ticker(ticker)

    # Map period strings to yfinance periods
    period_mapping = {
        "1mo": "1mo",
        "3mo": "3mo",
        "6mo": "6mo",
        "1y": "1y",
        "2y": "2y",
    }

    yf_period = period_mapping.get(period, "3mo")

    # Try NSE first, then BSE
    for ticker_symbol in [
        ticker_variants["yfinance_nse"],
        ticker_variants["yfinance_bse"],
    ]:
        try:
            stock = yf.Ticker(ticker_symbol)
            hist = stock.history(period=yf_period)

            if not hist.empty:
                # Calculate technical indicators
                indicators = {}

                # Calculate RSI (14-day)
                delta = hist["Close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                indicators["RSI"] = {
                    "current": round(rsi.iloc[-1], 2),
                    "value_14_days_ago": round(
                        rsi.iloc[-14] if len(rsi) >= 14 else rsi.iloc[0], 2
                    ),
                    "status": "Overbought"
                    if rsi.iloc[-1] > 70
                    else "Oversold"
                    if rsi.iloc[-1] < 30
                    else "Neutral",
                }

                # Calculate SMA (50-day and 200-day)
                sma_50 = hist["Close"].rolling(window=50).mean()
                sma_200 = hist["Close"].rolling(window=200).mean()
                indicators["SMA"] = {
                    "SMA_50": round(sma_50.iloc[-1], 2)
                    if not sma_50.isna().iloc[-1]
                    else None,
                    "SMA_200": round(sma_200.iloc[-1], 2)
                    if not sma_200.isna().iloc[-1]
                    else None,
                    "price_vs_sma50": "Above"
                    if not sma_50.isna().iloc[-1]
                    and hist["Close"].iloc[-1] > sma_50.iloc[-1]
                    else "Below",
                    "price_vs_sma200": "Above"
                    if not sma_200.isna().iloc[-1]
                    and hist["Close"].iloc[-1] > sma_200.iloc[-1]
                    else "Below",
                }

                # Calculate EMA (12-day and 26-day)
                ema_12 = hist["Close"].ewm(span=12).mean()
                ema_26 = hist["Close"].ewm(span=26).mean()
                indicators["EMA"] = {
                    "EMA_12": round(ema_12.iloc[-1], 2),
                    "EMA_26": round(ema_26.iloc[-1], 2),
                    "price_vs_ema12": "Above"
                    if hist["Close"].iloc[-1] > ema_12.iloc[-1]
                    else "Below",
                    "price_vs_ema26": "Above"
                    if hist["Close"].iloc[-1] > ema_26.iloc[-1]
                    else "Below",
                }

                # Calculate MACD
                macd_line = ema_12 - ema_26
                signal_line = macd_line.ewm(span=9).mean()
                histogram = macd_line - signal_line
                indicators["MACD"] = {
                    "MACD_line": round(macd_line.iloc[-1], 4),
                    "Signal_line": round(signal_line.iloc[-1], 4),
                    "Histogram": round(histogram.iloc[-1], 4),
                    "crossover_signal": "Bullish"
                    if macd_line.iloc[-1] > signal_line.iloc[-1]
                    and macd_line.iloc[-2] <= signal_line.iloc[-2]
                    else "Bearish"
                    if macd_line.iloc[-1] < signal_line.iloc[-1]
                    and macd_line.iloc[-2] >= signal_line.iloc[-2]
                    else "No crossover",
                }

                # Calculate additional momentum indicators
                # Rate of Change (ROC)
                roc_5 = (
                    (hist["Close"] - hist["Close"].shift(5)) / hist["Close"].shift(5)
                ) * 100
                roc_20 = (
                    (hist["Close"] - hist["Close"].shift(20)) / hist["Close"].shift(20)
                ) * 100
                indicators["Momentum"] = {
                    "ROC_5_days": round(roc_5.iloc[-1], 2)
                    if not roc_5.isna().iloc[-1]
                    else None,
                    "ROC_20_days": round(roc_20.iloc[-1], 2)
                    if not roc_20.isna().iloc[-1]
                    else None,
                    "momentum_status": "Strong"
                    if (roc_5.iloc[-1] > 2 and roc_20.iloc[-1] > 2)
                    else "Weak"
                    if (roc_5.iloc[-1] < -2 and roc_20.iloc[-1] < -2)
                    else "Moderate",
                }

                # Calculate Bollinger Bands for context
                bb_upper, bb_middle, bb_lower = calculate_bollinger_bands_simple(
                    hist["Close"], 20, 2
                )
                current_price = hist["Close"].iloc[-1]
                bb_position = ((current_price - bb_lower) / (bb_upper - bb_lower)) * 100
                indicators["Bollinger_Bands"] = {
                    "upper_band": round(bb_upper, 2),
                    "middle_band": round(bb_middle, 2),
                    "lower_band": round(bb_lower, 2),
                    "position": round(bb_position, 1),
                    "status": "Overbought"
                    if bb_position > 80
                    else "Oversold"
                    if bb_position < 20
                    else "Neutral",
                }

                return {
                    "ticker": ticker_variants["base"],
                    "exchange": "NSE" if ticker_symbol.endswith(".NS") else "BSE",
                    "current_price": round(current_price, 2),
                    "period": period,
                    "indicators": indicators,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "data_points": len(hist),
                }

        except Exception:
            continue

    raise ValueError(
        f"Could not fetch technical analysis data for ticker {ticker_variants['base']}"
    )


def calculate_bollinger_bands_simple(
    prices: pd.Series, window: int = 20, std_dev: float = 2
) -> Tuple[float, float, float]:
    """
    Simple Bollinger Bands calculation for technical indicators

    Args:
        prices: Series of closing prices
        window: Moving average window
        std_dev: Standard deviation multiplier

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    if len(prices) < window:
        # Not enough data, return current price for all bands
        current_price = prices.iloc[-1]
        return current_price, current_price, current_price

    middle_band = prices.rolling(window=window).mean().iloc[-1]
    std = prices.rolling(window=window).std().iloc[-1]
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)

    return upper_band, middle_band, lower_band


async def get_stock_actions(ticker: str) -> Dict[str, Any]:
    """
    Get corporate actions (dividends, stock splits) for a stock

    Args:
        ticker: Indian stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')

    Returns:
        Dictionary with dividend and stock split information
    """
    ticker_variants = normalize_ticker(ticker)

    # Try NSE first, then BSE
    stock_data = None
    exchange_used = None

    for exchange, ticker_symbol in [
        ("NSE", ticker_variants["yfinance_nse"]),
        ("BSE", ticker_variants["yfinance_bse"]),
    ]:
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            if info and "regularMarketPrice" in info:
                stock_data = stock
                exchange_used = exchange
                break
        except Exception:
            continue

    if not stock_data:
        raise ValueError(f"Could not fetch data for ticker {ticker_variants['base']}")

    # Get dividend data
    dividends = stock_data.dividends
    dividend_data = []

    if not dividends.empty:
        # Get last 10 dividend records
        recent_dividends = dividends.tail(10)
        for date, dividend_amount in recent_dividends.items():
            dividend_data.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "amount": round(dividend_amount, 2),
                    "amount_inr": round(
                        dividend_amount, 2
                    ),  # Assuming currency is INR for Indian stocks
                }
            )

    # Get stock split data
    splits = stock_data.splits
    split_data = []

    if not splits.empty:
        # Get all split records
        for date, split_ratio in splits.items():
            split_data.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "ratio": f"{int(split_ratio)}:{int(1 / split_ratio)}"
                    if split_ratio != 1
                    else "1:1",
                    "split_factor": round(split_ratio, 3),
                    "description": f"{split_ratio}:1 stock split",
                }
            )

    # Calculate dividend statistics
    total_dividends_5y = 0
    dividend_count = 0
    avg_annual_dividend = 0

    if not dividends.empty:
        # Filter dividends from last 5 years
        five_years_ago = pd.Timestamp.now(tz="Asia/Kolkata") - pd.DateOffset(years=5)
        recent_dividends = dividends[dividends.index >= five_years_ago]
        total_dividends_5y = recent_dividends.sum()
        dividend_count = len(recent_dividends)

        if dividend_count > 0:
            # Estimate annual dividend (assuming regular dividends)
            avg_annual_dividend = total_dividends_5y / (
                dividend_count / 4
            )  # Assuming quarterly on average

    # Get current dividend yield from info
    info = stock_data.info
    current_price = info.get("regularMarketPrice", info.get("currentPrice", 0))
    dividend_yield = (
        info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0
    )

    # Get current trailing annual dividend
    trailing_dividend = info.get("trailingAnnualDividendRate", 0)

    return {
        "ticker": ticker_variants["base"],
        "exchange": exchange_used,
        "current_price": current_price,
        "dividend_info": {
            "dividend_yield_percent": round(dividend_yield, 2),
            "trailing_annual_dividend": trailing_dividend,
            "estimated_annual_dividend": round(avg_annual_dividend, 2),
            "total_dividends_5years": round(total_dividends_5y, 2),
            "dividend_count_5years": dividend_count,
            "recent_dividends": dividend_data[-5:]
            if dividend_data
            else [],  # Last 5 dividends
            "dividend_frequency": "Quarterly"
            if dividend_count >= 15
            else "Semi-Annual"
            if dividend_count >= 8
            else "Annual"
            if dividend_count >= 4
            else "Irregular",
        },
        "stock_split_info": {
            "total_splits_count": len(split_data),
            "recent_splits": split_data,  # All split data (usually very few)
            "last_split_date": split_data[-1]["date"] if split_data else None,
            "last_split_ratio": split_data[-1]["ratio"] if split_data else None,
        },
        "dividend_analysis": {
            "is_dividend_paying": dividend_count > 0,
            "dividend_stability": "Stable"
            if dividend_count >= 15
            else "Irregular"
            if dividend_count >= 4
            else "None",
            "current_yield_status": "High"
            if dividend_yield > 3
            else "Moderate"
            if dividend_yield > 1
            else "Low",
            "payout_estimate_inr": round(current_price * (dividend_yield / 100), 2)
            if current_price and dividend_yield > 0
            else 0,
        },
        "last_updated": datetime.now().isoformat(),
    }


async def get_analyst_recommendations(ticker: str) -> Dict[str, Any]:
    """
    Get analyst recommendations and price targets for a stock

    Args:
        ticker: Indian stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')

    Returns:
        Dictionary with analyst recommendations and trends
    """
    ticker_variants = normalize_ticker(ticker)

    # Try NSE first, then BSE
    stock_data = None
    exchange_used = None

    for exchange, ticker_symbol in [
        ("NSE", ticker_variants["yfinance_nse"]),
        ("BSE", ticker_variants["yfinance_bse"]),
    ]:
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            if info and "regularMarketPrice" in info:
                stock_data = stock
                exchange_used = exchange
                break
        except Exception:
            continue

    if not stock_data:
        raise ValueError(f"Could not fetch data for ticker {ticker_variants['base']}")

    info = stock_data.info
    current_price = info.get("regularMarketPrice", info.get("currentPrice", 0))

    # Get current recommendation from Yahoo Finance
    current_recommendation = info.get("recommendationKey", "N/A")

    # Map Yahoo Finance recommendations to readable format
    recommendation_mapping = {
        "strong_buy": "Strong Buy",
        "buy": "Buy",
        "hold": "Hold",
        "sell": "Sell",
        "strong_sell": "Strong Sell",
    }

    current_recommendation_readable = recommendation_mapping.get(
        current_recommendation.lower(), current_recommendation
    )

    # Get analyst price targets if available
    price_targets = {
        "mean": info.get("targetMeanPrice", None),
        "high": info.get("targetHighPrice", None),
        "low": info.get("targetLowPrice", None),
        "median": info.get("targetMedianPrice", None),
        "current_price": current_price,
    }

    # Calculate price target upside/downside
    if price_targets["mean"]:
        upside_potential = (
            (price_targets["mean"] - current_price) / current_price
        ) * 100
        price_targets["upside_potential_percent"] = round(upside_potential, 2)
        price_targets["recommendation"] = (
            "Buy"
            if upside_potential > 10
            else "Hold"
            if upside_potential > -10
            else "Sell"
        )

    # Get number of analysts
    num_analysts = info.get("numberOfAnalystOpinions", None)

    # Try to get detailed recommendations from the recommendations data
    recommendations_data = None
    try:
        recommendations_data = stock_data.recommendations
    except Exception:
        # If recommendations data is not available, continue with basic info
        recommendations_data = None

    detailed_recommendations = {
        "strong_buy": 0,
        "buy": 0,
        "hold": 0,
        "sell": 0,
        "strong_sell": 0,
    }

    recommendations_trend = []

    if recommendations_data is not None and not recommendations_data.empty:
        # Process recommendations data if available
        # Get recent recommendations (last 10 records)
        recent_recommendations = recommendations_data.tail(10)

        # Count current recommendations
        if (
            "Firm" in recent_recommendations.columns
            and "To Grade" in recent_recommendations.columns
        ):
            # Group by recommendation type and count
            recommendation_counts = recent_recommendations["To Grade"].value_counts()

            for rec_type, count in recommendation_counts.items():
                rec_key = rec_type.lower().replace(" ", "_")
                if rec_key in detailed_recommendations:
                    detailed_recommendations[rec_key] = int(count)

            # Create trend data with dates
            if "Grade Date" in recent_recommendations.columns:
                for _, row in recent_recommendations.iterrows():
                    if pd.notna(row.get("Grade Date")) and pd.notna(
                        row.get("To Grade")
                    ):
                        recommendations_trend.append(
                            {
                                "date": row["Grade Date"].strftime("%Y-%m-%d")
                                if hasattr(row["Grade Date"], "strftime")
                                else str(row["Grade Date"]),
                                "firm": row.get("Firm", "Unknown"),
                                "recommendation": row.get("To Grade", "N/A"),
                                "action": "Initiated",  # Simplified - actual data might have upgrade/downgrade info
                            }
                        )

    # Create consensus summary
    total_recommendations = sum(detailed_recommendations.values())
    consensus_score = 0

    if total_recommendations > 0:
        # Calculate weighted consensus score (Strong Buy=5, Buy=4, Hold=3, Sell=2, Strong Sell=1)
        weights = {"strong_buy": 5, "buy": 4, "hold": 3, "sell": 2, "strong_sell": 1}
        for rec_type, count in detailed_recommendations.items():
            consensus_score += weights[rec_type] * count
        consensus_score = consensus_score / total_recommendations

    # Determine overall consensus
    if consensus_score >= 4.5:
        consensus_recommendation = "Strong Buy"
    elif consensus_score >= 3.5:
        consensus_recommendation = "Buy"
    elif consensus_score >= 2.5:
        consensus_recommendation = "Hold"
    elif consensus_score >= 1.5:
        consensus_recommendation = "Sell"
    else:
        consensus_recommendation = "Strong Sell"

    # Create analysis summary
    analysis_summary = {"strong_points": [], "caution_points": []}

    if price_targets.get("mean") and price_targets["mean"] > current_price * 1.1:
        analysis_summary["strong_points"].append(
            "Significant upside potential according to analyst price targets"
        )

    if num_analysts and num_analysts >= 10:
        analysis_summary["strong_points"].append(
            f"Well-covered by {num_analysts}+ analysts"
        )

    if current_recommendation_readable in ["Strong Buy", "Buy"]:
        analysis_summary["strong_points"].append(
            f"Current consensus: {current_recommendation_readable}"
        )

    if price_targets.get("mean") and price_targets["mean"] < current_price * 0.9:
        analysis_summary["caution_points"].append(
            "Price targets suggest potential downside"
        )

    if total_recommendations == 0:
        analysis_summary["caution_points"].append("Limited analyst coverage")

    return {
        "ticker": ticker_variants["base"],
        "exchange": exchange_used,
        "current_price": current_price,
        "current_recommendation": current_recommendation_readable,
        "analyst_count": num_analysts,
        "price_targets": price_targets,
        "recommendations_breakdown": detailed_recommendations,
        "consensus_score": round(consensus_score, 2)
        if total_recommendations > 0
        else None,
        "consensus_recommendation": consensus_recommendation
        if total_recommendations > 0
        else "Insufficient Data",
        "recommendations_trend": recommendations_trend[-5:]
        if recommendations_trend
        else [],  # Last 5 recommendations
        "total_recommendations_count": total_recommendations,
        "analysis_summary": analysis_summary,
        "last_updated": datetime.now().isoformat(),
    }


async def get_stock_holders(ticker: str) -> Dict[str, Any]:
    """
    Get major holders and institutional ownership information for a stock

    Args:
        ticker: Indian stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'INFY')

    Returns:
        Dictionary with major holders and institutional ownership data
    """
    ticker_variants = normalize_ticker(ticker)

    # Try NSE first, then BSE
    stock_data = None
    exchange_used = None

    for exchange, ticker_symbol in [
        ("NSE", ticker_variants["yfinance_nse"]),
        ("BSE", ticker_variants["yfinance_bse"]),
    ]:
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            if info and "regularMarketPrice" in info:
                stock_data = stock
                exchange_used = exchange
                break
        except Exception:
            continue

    if not stock_data:
        raise ValueError(f"Could not fetch data for ticker {ticker_variants['base']}")

    # Get major holders data
    major_holders_data = []
    try:
        major_holders = stock_data.major_holders
        if major_holders is not None and not major_holders.empty:
            # major_holders has breakdown as index and values as a single column
            for breakdown, value in major_holders.items():
                category = str(breakdown).strip()
                if "Percent" in category:
                    # Convert to percentage
                    percentage = float(value) * 100
                    display_value = f"{percentage:.2f}%"
                else:
                    percentage = float(value)
                    display_value = str(int(percentage))

                major_holders_data.append({
                    "category": category.replace("Percent", "").replace("Count", ""),
                    "percentage": round(percentage, 2),
                    "display_value": display_value
                })
    except Exception:
        major_holders_data = []

    # Get institutional holders data
    institutional_holders_data = []
    try:
        institutional_holders = stock_data.institutional_holders
        if institutional_holders is not None and not institutional_holders.empty:
            # Get top 10 institutional holders
            top_institutions = institutional_holders.head(10)
            for _, row in top_institutions.iterrows():
                institutional_holders_data.append(
                    {
                        "holder": row.get("Holder", "Unknown"),
                        "shares": int(row.get("Shares", 0)),
                        "date_reported": row.get("Date Reported", "N/A"),
                        "percentage_out": f"{row.get('% Out', 0)}%"
                        if row.get("% Out")
                        else "N/A",
                        "value": f"${row.get('Value', 0):,}"
                        if row.get("Value")
                        else "N/A",
                    }
                )
    except Exception:
        institutional_holders_data = []

    # Get insider holders data
    insider_holders_data = []
    try:
        insider_holders = stock_data.insider_holders
        if insider_holders is not None and not insider_holders.empty:
            # Get top 10 insider holders
            top_insiders = insider_holders.head(10)
            for _, row in top_insiders.iterrows():
                insider_holders_data.append(
                    {
                        "holder": row.get("Holder", "Unknown"),
                        "position": row.get("Position", "N/A"),
                        "shares": int(row.get("Latest", 0)),
                        "date_reported": row.get("Date Reported", "N/A"),
                        "percentage_out": f"{row.get('% Out', 0)}%"
                        if row.get("% Out")
                        else "N/A",
                    }
                )
    except Exception:
        insider_holders_data = []

    # Calculate ownership analysis
    total_institutional_holding = 0
    if institutional_holders_data:
        # Sum up institutional holdings from the percentage data
        for holder in institutional_holders_data:
            pct_str = holder["percentage_out"].replace("%", "")
            if pct_str != "N/A":
                try:
                    total_institutional_holding += float(pct_str)
                except (ValueError, TypeError):
                    pass

    # Get basic info for context
    info = stock_data.info
    market_cap = info.get("marketCap", 0)
    float_shares = info.get("floatShares", 0)

    # Analyze ownership structure
    ownership_analysis = {
        "high_institutional_confidence": total_institutional_holding > 50,
        "moderate_institutional_confidence": 25 < total_institutional_holding <= 50,
        "low_institutional_confidence": total_institutional_holding <= 25,
        "diverse_ownership": len(institutional_holders_data) >= 5,
        "concentrated_ownership": len(major_holders_data) > 0
        and any(h["percentage"] > 20 for h in major_holders_data),
    }

    # Create summary insights
    insights = {
        "ownership_strength": "Strong"
        if ownership_analysis["high_institutional_confidence"]
        else "Moderate"
        if ownership_analysis["moderate_institutional_confidence"]
        else "Limited",
        "key_investors": [],
        "ownership_distribution": "Distributed" not in "Concentrated",
        "investor_type": "Institutional"
        if total_institutional_holding > 50
        else "Mixed"
        if total_institutional_holding > 25
        else "Retail",
    }

    if institutional_holders_data:
        # Find top 3 institutional holders
        top_3_institutions = sorted(
            institutional_holders_data, key=lambda x: x["shares"], reverse=True
        )[:3]
        insights["key_investors"] = [holder["holder"] for holder in top_3_institutions]

    return {
        "ticker": ticker_variants["base"],
        "exchange": exchange_used,
        "market_cap": market_cap,
        "float_shares": float_shares,
        "major_holders": major_holders_data,
        "institutional_holders": institutional_holders_data,
        "insider_holders": insider_holders_data,
        "ownership_summary": {
            "total_institutional_holding_percent": round(
                total_institutional_holding, 2
            ),
            "number_of_institutional_holders": len(institutional_holders_data),
            "number_of_insider_holders": len(insider_holders_data),
            "number_of_major_holder_categories": len(major_holders_data),
            "top_institutional_holder": institutional_holders_data[0]["holder"]
            if institutional_holders_data
            else None,
        },
        "ownership_analysis": ownership_analysis,
        "insights": insights,
        "last_updated": datetime.now().isoformat(),
    }
