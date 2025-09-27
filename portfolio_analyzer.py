"""Portfolio analysis engine for performance evaluation and risk assessment.

This module provides comprehensive portfolio analytics including performance metrics,
risk assessment, benchmark comparisons, and compliance checking.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from market_research import MarketResearcher
from trading_script import load_latest_portfolio_state, PORTFOLIO_CSV


class PortfolioAnalyzer:
    """Comprehensive portfolio analysis and performance evaluation."""
    
    def __init__(self, finnhub_api_key: Optional[str] = None):
        """Initialize portfolio analyzer.
        
        Parameters
        ----------
        finnhub_api_key : str, optional
            Finnhub API key for market data
        """
        self.researcher = MarketResearcher(finnhub_api_key)
        self.max_market_cap = 300_000_000  # $300M micro-cap limit
        self.experiment_start = datetime(2025, 6, 27)
        self.baseline_equity = 100.0
        
    def load_portfolio_history(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """Load complete portfolio history from CSV.
        
        Parameters
        ----------
        csv_path : str, optional
            Path to portfolio CSV file. Defaults to main portfolio file.
            
        Returns
        -------
        pd.DataFrame
            Complete portfolio history
        """
        if csv_path is None:
            # Try to use current PORTFOLIO_CSV path (which might be set by set_data_dir)
            try:
                from trading_script import PORTFOLIO_CSV
                csv_path = str(PORTFOLIO_CSV)
            except:
                # Fallback to Start Your Own directory
                csv_path = str(Path(__file__).parent / "Start Your Own" / "chatgpt_portfolio_update.csv")
        
        try:
            df = pd.read_csv(csv_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            print(f"Error loading portfolio history: {e}")
            return pd.DataFrame()
    
    def get_current_portfolio_state(self) -> Tuple[List[Dict], float]:
        """Get current portfolio positions and cash balance.
        
        Returns
        -------
        tuple
            (portfolio_positions, cash_balance)
        """
        try:
            # Try to use current PORTFOLIO_CSV path (which might be set by set_data_dir)
            try:
                from trading_script import PORTFOLIO_CSV
                csv_path = str(PORTFOLIO_CSV)
            except:
                # Fallback to Start Your Own directory
                csv_path = str(Path(__file__).parent / "Start Your Own" / "chatgpt_portfolio_update.csv")
            
            portfolio_data, cash_balance = load_latest_portfolio_state()
            return portfolio_data, cash_balance
        except Exception as e:
            print(f"Error loading current portfolio: {e}")
            return [], 100.0
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics.
        
        Returns
        -------
        dict
            Performance metrics including returns, ratios, and drawdowns
        """
        df = self.load_portfolio_history()
        
        if df.empty:
            return {"error": "No portfolio history available"}
        
        # Get TOTAL rows for equity tracking
        totals = df[df['Ticker'] == 'TOTAL'].copy()
        totals = totals.sort_values('Date')
        
        if len(totals) < 2:
            return {"error": "Insufficient data for performance calculation"}
        
        # Calculate metrics
        equity_series = totals['Total Equity'].astype(float)
        dates = pd.to_datetime(totals['Date'])
        
        # Basic returns
        total_return = (equity_series.iloc[-1] - self.baseline_equity) / self.baseline_equity
        current_equity = equity_series.iloc[-1]
        
        # Daily returns
        daily_returns = equity_series.pct_change().dropna()
        
        # Time metrics
        trading_days = len(totals) - 1
        calendar_days = (dates.iloc[-1] - dates.iloc[0]).days
        
        # Annualized return (assuming 252 trading days per year)
        if trading_days > 0:
            annualized_return = (1 + total_return) ** (252 / trading_days) - 1
        else:
            annualized_return = 0
        
        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Sharpe ratio (assuming 4.5% risk-free rate)
        rf_rate = 0.045
        if volatility > 0:
            sharpe_ratio = (annualized_return - rf_rate) / volatility
        else:
            sharpe_ratio = 0
        
        # Sortino ratio (downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = negative_returns.std() * np.sqrt(252)
            if downside_deviation > 0:
                sortino_ratio = (annualized_return - rf_rate) / downside_deviation
            else:
                sortino_ratio = 0
        else:
            sortino_ratio = float('inf') if annualized_return > rf_rate else 0
        
        # Maximum drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_days = len(daily_returns[daily_returns > 0])
        total_trading_days = len(daily_returns)
        win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0
        
        return {
            "current_equity": current_equity,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "annualized_return": annualized_return,
            "annualized_return_pct": annualized_return * 100,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "win_rate": win_rate,
            "win_rate_pct": win_rate * 100,
            "trading_days": trading_days,
            "calendar_days": calendar_days,
            "daily_avg_return": daily_returns.mean(),
            "best_day": daily_returns.max(),
            "worst_day": daily_returns.min(),
        }
    
    def compare_to_benchmark(self, benchmark_ticker: str = "SPY") -> Dict:
        """Compare portfolio performance to benchmark.
        
        Parameters
        ----------
        benchmark_ticker : str, optional
            Benchmark ticker symbol (default: SPY for S&P 500)
            
        Returns
        -------
        dict
            Benchmark comparison metrics
        """
        df = self.load_portfolio_history()
        
        if df.empty:
            return {"error": "No portfolio data for comparison"}
        
        # Get portfolio dates
        totals = df[df['Ticker'] == 'TOTAL'].copy()
        start_date = pd.to_datetime(totals['Date']).min()
        end_date = pd.to_datetime(totals['Date']).max()
        
        # Get benchmark data
        try:
            import time
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            
            candles = self.researcher.client.stock_candles(
                benchmark_ticker, 'D', start_timestamp, end_timestamp
            )
            
            if not candles or candles.get('s') != 'ok':
                return {"error": f"Could not fetch {benchmark_ticker} data"}
            
            # Calculate benchmark return
            benchmark_start = candles['c'][0]
            benchmark_end = candles['c'][-1]
            benchmark_return = (benchmark_end - benchmark_start) / benchmark_start
            
            # Portfolio return
            portfolio_return = self.calculate_performance_metrics().get("total_return", 0)
            
            # Alpha calculation
            alpha = portfolio_return - benchmark_return
            
            return {
                "benchmark_ticker": benchmark_ticker,
                "benchmark_start_price": benchmark_start,
                "benchmark_end_price": benchmark_end,
                "benchmark_return": benchmark_return,
                "benchmark_return_pct": benchmark_return * 100,
                "portfolio_return": portfolio_return,
                "portfolio_return_pct": portfolio_return * 100,
                "alpha": alpha,
                "alpha_pct": alpha * 100,
                "outperforming": alpha > 0
            }
            
        except Exception as e:
            return {"error": f"Benchmark comparison failed: {str(e)}"}
    
    def analyze_current_positions(self) -> List[Dict]:
        """Analyze each current portfolio position.
        
        Returns
        -------
        list
            Analysis of each position including compliance and metrics
        """
        positions, cash = self.get_current_portfolio_state()
        
        if not positions:
            return []
        
        position_analyses = []
        
        for position in positions:
            ticker = position.get('ticker', '')
            shares = position.get('shares', 0)
            cost_basis = position.get('buy_price', 0)
            stop_loss = position.get('stop_loss', 0)
            
            # Get current market data
            fundamentals = self.researcher.get_stock_fundamentals(ticker)
            
            current_price = fundamentals.get('current_price', 0)
            market_cap = fundamentals.get('market_cap', 0)
            
            # Calculate position metrics
            current_value = shares * current_price
            unrealized_pnl = (current_price - cost_basis) * shares
            unrealized_return = (current_price - cost_basis) / cost_basis if cost_basis > 0 else 0
            
            # Risk assessment
            distance_to_stop = (current_price - stop_loss) / current_price if current_price > 0 and stop_loss > 0 else 0
            
            analysis = {
                "ticker": ticker,
                "shares": shares,
                "cost_basis": cost_basis,
                "current_price": current_price,
                "current_value": current_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_return": unrealized_return,
                "unrealized_return_pct": unrealized_return * 100,
                "stop_loss": stop_loss,
                "distance_to_stop_pct": distance_to_stop * 100,
                "market_cap": market_cap,
                "is_micro_cap": self.researcher.is_micro_cap(market_cap),
                "micro_cap_compliant": fundamentals.get('micro_cap_eligible', False),
                "company_name": fundamentals.get('company_name', ''),
                "industry": fundamentals.get('industry', ''),
                "compliance_issues": []
            }
            
            # Check compliance
            if not analysis["is_micro_cap"]:
                analysis["compliance_issues"].append("Market cap exceeds $300M limit")
            
            if fundamentals.get("country") != "US":
                analysis["compliance_issues"].append("Not a US-listed stock")
            
            if current_price < 1.0:
                analysis["compliance_issues"].append("Penny stock (price < $1)")
            
            position_analyses.append(analysis)
        
        return position_analyses
    
    def assess_portfolio_risk(self) -> Dict:
        """Assess overall portfolio risk and diversification.
        
        Returns
        -------
        dict
            Risk assessment metrics
        """
        positions = self.analyze_current_positions()
        _, cash = self.get_current_portfolio_state()
        performance = self.calculate_performance_metrics()
        
        if not positions:
            return {"error": "No positions to analyze"}
        
        # Concentration analysis
        total_value = sum(pos['current_value'] for pos in positions)
        total_equity = total_value + cash
        
        position_weights = []
        for pos in positions:
            weight = pos['current_value'] / total_equity if total_equity > 0 else 0
            position_weights.append({
                "ticker": pos['ticker'],
                "weight": weight,
                "weight_pct": weight * 100
            })
        
        # Diversification metrics
        num_positions = len(positions)
        max_position_weight = max(weight['weight'] for weight in position_weights) if position_weights else 0
        
        # Industry concentration
        industries = {}
        for pos in positions:
            industry = pos.get('industry', 'Unknown')
            industries[industry] = industries.get(industry, 0) + pos['current_value']
        
        # Compliance check
        non_compliant = [pos for pos in positions if pos['compliance_issues']]
        compliance_rate = (num_positions - len(non_compliant)) / num_positions if num_positions > 0 else 1
        
        # Risk score (1-10, higher = riskier)
        risk_score = 1
        
        if max_position_weight > 0.5:  # More than 50% in one position
            risk_score += 3
        elif max_position_weight > 0.3:  # More than 30% in one position
            risk_score += 2
        
        if num_positions < 3:  # Too few positions
            risk_score += 2
        
        if compliance_rate < 1.0:  # Compliance issues
            risk_score += 2
        
        if performance.get('volatility', 0) > 0.3:  # High volatility
            risk_score += 1
        
        risk_score = min(risk_score, 10)  # Cap at 10
        
        return {
            "num_positions": num_positions,
            "max_position_weight": max_position_weight,
            "max_position_weight_pct": max_position_weight * 100,
            "position_weights": position_weights,
            "cash_weight": cash / total_equity if total_equity > 0 else 0,
            "cash_weight_pct": (cash / total_equity * 100) if total_equity > 0 else 0,
            "industry_concentration": industries,
            "compliance_issues": len(non_compliant),
            "compliance_rate": compliance_rate,
            "compliance_rate_pct": compliance_rate * 100,
            "non_compliant_positions": [pos['ticker'] for pos in non_compliant],
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score),
            "diversification_score": self._calculate_diversification_score(position_weights)
        }
    
    def _get_risk_level(self, risk_score: int) -> str:
        """Convert numeric risk score to text level."""
        if risk_score <= 3:
            return "LOW"
        elif risk_score <= 6:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _calculate_diversification_score(self, position_weights: List[Dict]) -> float:
        """Calculate diversification score using Herfindahl index."""
        if not position_weights:
            return 0
        
        # Herfindahl index (sum of squared weights)
        hhi = sum(weight['weight'] ** 2 for weight in position_weights)
        
        # Diversification score (inverse of HHI, normalized)
        max_hhi = 1.0  # Maximum concentration (all in one position)
        min_hhi = 1 / len(position_weights)  # Perfect diversification
        
        if max_hhi > min_hhi:
            diversification = (max_hhi - hhi) / (max_hhi - min_hhi)
        else:
            diversification = 1.0
        
        return max(0, min(1, diversification))
    
    def generate_portfolio_report(self) -> Dict:
        """Generate comprehensive portfolio analysis report.
        
        Returns
        -------
        dict
            Complete portfolio analysis report
        """
        return {
            "report_date": datetime.now().isoformat(),
            "performance": self.calculate_performance_metrics(),
            "benchmark_comparison": self.compare_to_benchmark(),
            "current_positions": self.analyze_current_positions(),
            "risk_assessment": self.assess_portfolio_risk(),
            "experiment_status": {
                "start_date": self.experiment_start.strftime("%Y-%m-%d"),
                "current_date": datetime.now().strftime("%Y-%m-%d"),
                "days_elapsed": (datetime.now() - self.experiment_start).days,
                "days_remaining": (datetime(2025, 12, 27) - datetime.now()).days
            }
        }
    
    def save_report(self, report: Dict, filename: Optional[str] = None):
        """Save analysis report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_analysis_{timestamp}.json"
        
        reports_dir = Path(__file__).parent / "analysis_reports"
        reports_dir.mkdir(exist_ok=True)
        
        with open(reports_dir / filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to: {reports_dir / filename}")


if __name__ == "__main__":
    # Example usage
    analyzer = PortfolioAnalyzer()
    
    # Generate comprehensive report
    report = analyzer.generate_portfolio_report()
    
    print("=== Portfolio Analysis Report ===")
    
    # Performance summary
    if "performance" in report and "error" not in report["performance"]:
        perf = report["performance"]
        print(f"Current Equity: ${perf['current_equity']:,.2f}")
        print(f"Total Return: {perf['total_return_pct']:+.2f}%")
        print(f"Annualized Return: {perf['annualized_return_pct']:+.2f}%")
        print(f"Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {perf['max_drawdown_pct']:.2f}%")
    
    # Position summary
    if "current_positions" in report:
        print(f"\nPositions: {len(report['current_positions'])}")
        for pos in report["current_positions"]:
            print(f"- {pos['ticker']}: ${pos['current_value']:,.2f} ({pos['unrealized_return_pct']:+.1f}%)")
            if pos['compliance_issues']:
                print(f"  Issues: {', '.join(pos['compliance_issues'])}")
    
    # Risk summary
    if "risk_assessment" in report and "error" not in report["risk_assessment"]:
        risk = report["risk_assessment"]
        print(f"\nRisk Level: {risk['risk_level']} (Score: {risk['risk_score']}/10)")
        print(f"Compliance Rate: {risk['compliance_rate_pct']:.1f}%")
    
    # Save report
    analyzer.save_report(report)