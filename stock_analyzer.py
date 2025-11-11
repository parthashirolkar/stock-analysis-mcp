"""
Indian Stock Market Data Analyzer

Core functionality for fetching Indian stock data from BSE/NSE.
Optimized for MCP server usage with FastMCP.
"""

import yfinance as yf
import re
from datetime import datetime
from typing import Dict, Any, List


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
