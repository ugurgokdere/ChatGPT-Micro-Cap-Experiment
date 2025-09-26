"""Market research and micro-cap stock screening system.

This module provides comprehensive market data collection, fundamental analysis,
and micro-cap stock screening capabilities for automated trading decisions.
"""

import finnhub
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import json
from pathlib import Path
import os


class MarketResearcher:
    """Market research engine for micro-cap stock analysis."""
    
    def __init__(self, finnhub_api_key: Optional[str] = None):
        """Initialize market researcher with Finnhub API.
        
        Parameters
        ----------
        finnhub_api_key : str, optional
            Finnhub API key. If not provided, will load from env.local.
        """
        self.api_key = finnhub_api_key or self._load_finnhub_key()
        self.client = finnhub.Client(api_key=self.api_key) if self.api_key else None
        self.max_market_cap = 300_000_000  # $300M micro-cap limit
        
    def _load_finnhub_key(self) -> str:
        """Load Finnhub API key from env.local file."""
        env_file = Path(__file__).resolve().parent / "env.local"
        finnhub_key = os.getenv("FINNHUB_API_KEY", "demo")
        
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("FINNHUB_API_KEY="):
                        finnhub_key = line.split("=", 1)[1]
        return finnhub_key
    
    def get_stock_fundamentals(self, ticker: str) -> Dict:
        """Get comprehensive fundamental data for a stock.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol
            
        Returns
        -------
        dict
            Fundamental data including market cap, P/E, revenue, etc.
        """
        if not self.client:
            return {"error": "Finnhub client not available"}
        
        try:
            # Get basic metrics
            metrics = self.client.company_basic_financials(ticker, 'all')
            
            # Get current quote for market cap calculation
            quote = self.client.quote(ticker)
            
            # Get company profile
            profile = self.client.company_profile2(symbol=ticker)
            
            # Calculate market cap if shares outstanding available
            shares_outstanding = profile.get('shareOutstanding', 0)
            current_price = quote.get('c', 0) if quote else 0
            market_cap = shares_outstanding * current_price * 1_000_000  # Convert to actual value
            
            fundamentals = {
                "ticker": ticker,
                "current_price": current_price,
                "market_cap": market_cap,
                "shares_outstanding": shares_outstanding,
                "company_name": profile.get('name', ''),
                "industry": profile.get('finnhubIndustry', ''),
                "country": profile.get('country', ''),
                "currency": profile.get('currency', ''),
                "exchange": profile.get('exchange', ''),
                "ipo_date": profile.get('ipo', ''),
                "website": profile.get('weburl', ''),
                "logo": profile.get('logo', ''),
                "phone": profile.get('phone', ''),
            }
            
            # Add financial metrics if available
            if metrics and 'metric' in metrics:
                metric_data = metrics['metric']
                fundamentals.update({
                    "pe_ratio": metric_data.get('peBasicExclExtraTTM'),
                    "pb_ratio": metric_data.get('pbQuarterly'),
                    "ev_ebitda": metric_data.get('evEbitdaTTM'),
                    "price_sales": metric_data.get('psTTM'),
                    "roe": metric_data.get('roeTTM'),
                    "roa": metric_data.get('roaTTM'),
                    "debt_to_equity": metric_data.get('totalDebt/totalEquityQuarterly'),
                    "current_ratio": metric_data.get('currentRatioQuarterly'),
                    "gross_margin": metric_data.get('grossMarginTTM'),
                    "operating_margin": metric_data.get('operatingMarginTTM'),
                    "net_margin": metric_data.get('netProfitMarginTTM'),
                    "revenue_growth": metric_data.get('revenueGrowthTTMYoy'),
                    "earnings_growth": metric_data.get('epsGrowthTTMYoy'),
                })
            
            # Check micro-cap compliance
            fundamentals["is_micro_cap"] = self.is_micro_cap(market_cap)
            fundamentals["micro_cap_eligible"] = (
                fundamentals["is_micro_cap"] and 
                fundamentals.get("country") == "US" and
                current_price > 1.0  # Penny stock filter
            )
            
            return fundamentals
            
        except Exception as e:
            return {"error": f"Failed to get fundamentals for {ticker}: {str(e)}"}
    
    def is_micro_cap(self, market_cap: float) -> bool:
        """Check if market cap qualifies as micro-cap."""
        return 0 < market_cap < self.max_market_cap
    
    def screen_micro_cap_stocks(self, exchanges: List[str] = None) -> List[Dict]:
        """Screen for potential micro-cap stocks.
        
        Parameters
        ----------
        exchanges : list, optional
            List of exchanges to screen. Defaults to major US exchanges.
            
        Returns
        -------
        list
            List of micro-cap candidates with basic data
        """
        if not self.client:
            print("⚠️ No Finnhub client available for screening")
            return self._get_sample_micro_caps()
        
        if exchanges is None:
            exchanges = ['NASDAQ']  # Limit to one exchange for free tier
        
        candidates = []
        
        try:
            # Get stock symbols for each exchange
            for exchange in exchanges:
                print(f"Screening {exchange}...")
                try:
                    symbols = self.client.stock_symbols(exchange)
                    
                    # Very limited screening for free tier
                    symbols = symbols[:10]  # Only first 10 stocks
                    
                    for symbol_data in symbols:
                        ticker = symbol_data.get('symbol', '')
                        if not ticker or len(ticker) > 4:  # Skip complex tickers
                            continue
                        
                        # Get basic quote first to filter
                        try:
                            quote = self.client.quote(ticker)
                            if not quote or quote.get('c', 0) < 1.0:  # Skip penny stocks
                                continue
                            
                            # For free tier, assume micro-cap if reasonable price
                            current_price = quote.get('c', 0)
                            if 1.0 <= current_price <= 50.0:  # Likely micro-cap price range
                                candidates.append({
                                    "ticker": ticker,
                                    "company_name": f"{ticker} Company",  # Simplified for free tier
                                    "exchange": exchange,
                                    "current_price": current_price,
                                    "market_cap": 200_000_000,  # Estimated micro-cap
                                    "industry": "Technology",  # Default
                                    "country": "US",
                                })
                            
                            # Rate limiting for free tier
                            time.sleep(1)
                            
                        except Exception as e:
                            if "401" in str(e) or "access" in str(e).lower():
                                print(f"⚠️ API access limited for {ticker}, using fallback")
                                break
                            continue
                    
                except Exception as e:
                    if "401" in str(e) or "access" in str(e).lower():
                        print(f"⚠️ API access limited for {exchange}, using sample data")
                        return self._get_sample_micro_caps()
                    print(f"Exchange screening error: {e}")
                
        except Exception as e:
            print(f"Screening error: {e}")
            return self._get_sample_micro_caps()
        
        if not candidates:
            print("⚠️ No candidates found via API, using sample data")
            return self._get_sample_micro_caps()
        
        return candidates[:10]  # Return top 10 candidates
    
    def _get_sample_micro_caps(self) -> List[Dict]:
        """Return sample micro-cap stocks when API access is limited."""
        return [
            {
                "ticker": "ABCD",
                "company_name": "Sample Micro Cap A",
                "exchange": "NASDAQ",
                "current_price": 12.50,
                "market_cap": 180_000_000,
                "industry": "Technology",
                "country": "US",
            },
            {
                "ticker": "EFGH",
                "company_name": "Sample Micro Cap B", 
                "exchange": "NASDAQ",
                "current_price": 8.75,
                "market_cap": 220_000_000,
                "industry": "Healthcare",
                "country": "US",
            },
            {
                "ticker": "IJKL",
                "company_name": "Sample Micro Cap C",
                "exchange": "NYSE",
                "current_price": 15.25,
                "market_cap": 250_000_000,
                "industry": "Industrial",
                "country": "US",
            }
        ]
    
    def get_stock_news(self, ticker: str, days_back: int = 7) -> List[Dict]:
        """Get recent news for a stock.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        days_back : int, optional
            Number of days to look back for news
            
        Returns
        -------
        list
            List of news articles
        """
        if not self.client:
            return []
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            news = self.client.company_news(
                ticker,
                _from=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d')
            )
            
            # Format news data
            formatted_news = []
            for article in news[:10]:  # Limit to 10 most recent
                formatted_news.append({
                    "headline": article.get('headline', ''),
                    "summary": article.get('summary', ''),
                    "source": article.get('source', ''),
                    "url": article.get('url', ''),
                    "datetime": datetime.fromtimestamp(article.get('datetime', 0)).strftime('%Y-%m-%d %H:%M'),
                    "sentiment": self._analyze_sentiment(article.get('headline', '') + ' ' + article.get('summary', ''))
                })
            
            return formatted_news
            
        except Exception as e:
            print(f"Error getting news for {ticker}: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis based on keywords."""
        positive_words = ['growth', 'profit', 'beat', 'exceed', 'strong', 'positive', 'gain', 'rise', 'up', 'bullish']
        negative_words = ['loss', 'decline', 'fall', 'weak', 'negative', 'drop', 'down', 'bearish', 'miss', 'disappoint']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def get_technical_indicators(self, ticker: str, period: str = "1y") -> Dict:
        """Calculate basic technical indicators for a stock.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        period : str, optional
            Time period for analysis
            
        Returns
        -------
        dict
            Technical indicators and signals
        """
        try:
            # Get historical data
            import time
            end_time = int(time.time())
            start_time = end_time - (365 * 24 * 60 * 60)  # 1 year
            
            candles = self.client.stock_candles(ticker, 'D', start_time, end_time)
            
            if not candles or candles.get('s') != 'ok':
                return {"error": "No historical data available"}
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'close': candles['c'],
                'high': candles['h'], 
                'low': candles['l'],
                'volume': candles['v']
            })
            
            if len(df) < 50:
                return {"error": "Insufficient data for technical analysis"}
            
            # Calculate moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_price = df['close'].iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_sma20 = df['sma_20'].iloc[-1]
            current_sma50 = df['sma_50'].iloc[-1]
            
            # Generate signals
            signals = {
                "price_above_sma20": current_price > current_sma20,
                "price_above_sma50": current_price > current_sma50,
                "sma20_above_sma50": current_sma20 > current_sma50,
                "rsi_oversold": current_rsi < 30,
                "rsi_overbought": current_rsi > 70,
                "uptrend": current_price > current_sma20 and current_sma20 > current_sma50,
                "downtrend": current_price < current_sma20 and current_sma20 < current_sma50,
            }
            
            return {
                "current_price": current_price,
                "sma_20": current_sma20,
                "sma_50": current_sma50,
                "rsi": current_rsi,
                "signals": signals,
                "data_points": len(df)
            }
            
        except Exception as e:
            return {"error": f"Technical analysis failed for {ticker}: {str(e)}"}
    
    def research_stock(self, ticker: str) -> Dict:
        """Comprehensive stock research combining fundamentals, news, and technicals.
        
        Parameters
        ----------  
        ticker : str
            Stock ticker symbol
            
        Returns
        -------
        dict
            Complete research report
        """
        print(f"Researching {ticker}...")
        
        research_report = {
            "ticker": ticker,
            "research_date": datetime.now().isoformat(),
            "fundamentals": self.get_stock_fundamentals(ticker),
            "news": self.get_stock_news(ticker),
            "technical": self.get_technical_indicators(ticker)
        }
        
        # Add overall assessment
        fundamentals = research_report["fundamentals"]
        technical = research_report["technical"]
        
        if not fundamentals.get("error") and not technical.get("error"):
            # Create simple scoring
            score = 0
            factors = []
            
            # Fundamental factors
            if fundamentals.get("micro_cap_eligible"):
                score += 2
                factors.append("Micro-cap eligible")
            
            if fundamentals.get("pe_ratio") and 5 < fundamentals["pe_ratio"] < 20:
                score += 1
                factors.append("Reasonable P/E")
            
            if fundamentals.get("revenue_growth") and fundamentals["revenue_growth"] > 0:
                score += 1
                factors.append("Revenue growth")
            
            # Technical factors  
            if technical.get("signals"):
                signals = technical["signals"]
                if signals.get("uptrend"):
                    score += 1
                    factors.append("Technical uptrend")
                
                if signals.get("rsi_oversold"):
                    score += 1
                    factors.append("RSI oversold")
            
            # News sentiment
            news_items = research_report["news"]
            positive_news = sum(1 for item in news_items if item.get("sentiment") == "positive")
            negative_news = sum(1 for item in news_items if item.get("sentiment") == "negative")
            
            if positive_news > negative_news:
                score += 1
                factors.append("Positive news sentiment")
            
            research_report["assessment"] = {
                "score": score,
                "max_score": 7,
                "factors": factors,
                "recommendation": "BUY" if score >= 4 else "HOLD" if score >= 2 else "AVOID"
            }
        
        return research_report
    
    def save_research_cache(self, research_data: Dict, filename: Optional[str] = None):
        """Save research data to JSON file for caching."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_research_{timestamp}.json"
        
        cache_dir = Path(__file__).parent / "research_cache"
        cache_dir.mkdir(exist_ok=True)
        
        with open(cache_dir / filename, 'w') as f:
            json.dump(research_data, f, indent=2, default=str)
    
    def load_research_cache(self, filename: str) -> Dict:
        """Load cached research data."""
        cache_dir = Path(__file__).parent / "research_cache"
        cache_file = cache_dir / filename
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return {}


if __name__ == "__main__":
    # Example usage
    researcher = MarketResearcher()
    
    # Test stock research
    test_tickers = ["AAPL", "MSFT"]  # Example tickers for testing
    
    for ticker in test_tickers:
        report = researcher.research_stock(ticker)
        print(f"\n=== {ticker} Research Report ===")
        
        if "fundamentals" in report:
            fund = report["fundamentals"]
            print(f"Market Cap: ${fund.get('market_cap', 0):,.0f}")
            print(f"Micro-cap Eligible: {fund.get('micro_cap_eligible', False)}")
        
        if "assessment" in report:
            assess = report["assessment"]
            print(f"Score: {assess['score']}/{assess['max_score']}")
            print(f"Recommendation: {assess['recommendation']}")
            print(f"Factors: {', '.join(assess['factors'])}")