"""N8N Automation Wrapper for Trading Scripts

This script provides a command-line interface for n8n to execute
trading analysis and return structured JSON results for email notifications.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from interactive_research_analyzer import InteractiveResearchAnalyzer
from daily_trade_logger import DailyTradeLogger
from portfolio_analyzer import PortfolioAnalyzer
from market_research import MarketResearcher


class N8NTradingAutomation:
    """Automation wrapper for n8n workflow integration."""
    
    def __init__(self):
        """Initialize automation wrapper."""
        self.analyzer = InteractiveResearchAnalyzer()
        self.logger = DailyTradeLogger()
        self.portfolio_analyzer = PortfolioAnalyzer()
        self.market_researcher = MarketResearcher()
        
    def analyze_watchlist(self, tickers: List[str]) -> Dict:
        """Analyze a predefined watchlist of stocks.
        
        Parameters
        ----------
        tickers : list
            List of stock tickers to analyze
            
        Returns
        -------
        dict
            Analysis results with buy/sell/hold recommendations
        """
        try:
            # Get detailed analysis for each ticker
            detailed_analysis = self.analyzer.get_detailed_analysis(tickers)
            
            # Generate AI recommendations
            ai_analysis = self.analyzer.generate_ai_recommendations(tickers, detailed_analysis)
            
            # Generate final recommendations
            recommendations = self.analyzer.generate_final_recommendations(
                tickers, detailed_analysis, ai_analysis
            )
            
            # Format for email notification
            email_content = self._format_email_content(recommendations)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "tickers_analyzed": tickers,
                "recommendations": recommendations,
                "email_content": email_content,
                "summary": {
                    "buy_count": len(recommendations.get("buy_candidates", [])),
                    "sell_count": len(recommendations.get("sell_recommendations", [])),
                    "hold_count": len(recommendations.get("hold_positions", []))
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status and performance.
        
        Returns
        -------
        dict
            Current portfolio state and metrics
        """
        try:
            positions, cash = self.portfolio_analyzer.get_current_portfolio_state()
            
            total_value = sum(pos.get('shares', 0) * pos.get('buy_price', 0) for pos in positions)
            total_equity = total_value + cash
            
            # Get recent trades
            trade_history = self.logger.load_trade_history()
            recent_trades = trade_history[-5:] if trade_history else []
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "portfolio": {
                    "positions": len(positions),
                    "total_invested": total_value,
                    "cash_available": cash,
                    "total_equity": total_equity,
                    "holdings": [
                        {
                            "ticker": pos['ticker'],
                            "shares": pos['shares'],
                            "cost_basis": pos['buy_price'],
                            "current_value": pos['shares'] * pos['buy_price']
                        }
                        for pos in positions
                    ]
                },
                "recent_trades": recent_trades
            }
            
        except Exception as e:
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _format_email_content(self, recommendations: Dict) -> Dict:
        """Format recommendations for email notification.
        
        Parameters
        ----------
        recommendations : dict
            Trading recommendations
            
        Returns
        -------
        dict
            Formatted email content with subject and body
        """
        # Generate email subject
        buy_count = len(recommendations.get("buy_candidates", []))
        sell_count = len(recommendations.get("sell_recommendations", []))
        
        if buy_count > 0 or sell_count > 0:
            subject = f"🚨 ACTION REQUIRED: {buy_count} BUY, {sell_count} SELL signals"
        else:
            subject = f"📊 Portfolio Update: All positions on HOLD"
        
        # Generate email body
        body_html = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2>Trading Analysis Report</h2>
            <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            
            <h3>📈 BUY Recommendations ({buy_count})</h3>
        """
        
        if recommendations.get("buy_candidates"):
            body_html += "<ul>"
            for candidate in recommendations["buy_candidates"]:
                body_html += f"""
                <li>
                    <strong>{candidate['ticker']}</strong> @ ${candidate['current_price']:.2f}<br>
                    Confidence: {candidate.get('confidence', 'MEDIUM')}<br>
                    Reason: {candidate['reason']}<br>
                """
                if candidate.get('key_factors'):
                    body_html += f"Key Factors: {', '.join(candidate['key_factors'])}<br>"
                body_html += "</li>"
            body_html += "</ul>"
        else:
            body_html += "<p>No buy recommendations at this time.</p>"
        
        body_html += f"""
            <h3>📉 SELL Recommendations ({sell_count})</h3>
        """
        
        if recommendations.get("sell_recommendations"):
            body_html += "<ul>"
            for sell in recommendations["sell_recommendations"]:
                body_html += f"""
                <li>
                    <strong>{sell['ticker']}</strong><br>
                    Reason: {sell['reason']}<br>
                    Confidence: {sell.get('confidence', 'MEDIUM')}
                </li>
                """
            body_html += "</ul>"
        else:
            body_html += "<p>No sell recommendations at this time.</p>"
        
        # Add hold positions summary
        hold_count = len(recommendations.get("hold_positions", []))
        body_html += f"""
            <h3>⏸ HOLD Positions ({hold_count})</h3>
        """
        
        if recommendations.get("hold_positions"):
            body_html += "<ul>"
            for hold in recommendations["hold_positions"][:5]:  # Show first 5
                body_html += f"""
                <li>
                    <strong>{hold['ticker']}</strong> @ ${hold['current_price']:.2f}<br>
                    {hold['reason']}
                </li>
                """
            if hold_count > 5:
                body_html += f"<li><em>...and {hold_count - 5} more</em></li>"
            body_html += "</ul>"
        
        # Add AI analysis excerpt if available
        ai_analysis = recommendations.get("ai_analysis", {})
        if ai_analysis and not ai_analysis.get("error"):
            analysis_text = ai_analysis.get("analysis", "")
            if analysis_text:
                # Extract first paragraph of AI analysis
                first_para = analysis_text.split('\n\n')[0] if analysis_text else ""
                if first_para:
                    body_html += f"""
                    <h3>🤖 AI Analysis Summary</h3>
                    <p>{first_para}</p>
                    """
        
        body_html += """
            <hr>
            <p style="color: #666; font-size: 12px;">
                This is an automated trading analysis. Always verify recommendations before trading.
            </p>
        </body>
        </html>
        """
        
        # Plain text version
        body_text = f"""Trading Analysis Report
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

BUY Recommendations ({buy_count}):
"""
        
        if recommendations.get("buy_candidates"):
            for candidate in recommendations["buy_candidates"]:
                body_text += f"- {candidate['ticker']} @ ${candidate['current_price']:.2f} ({candidate.get('confidence', 'MEDIUM')})\n"
        else:
            body_text += "None\n"
        
        body_text += f"\nSELL Recommendations ({sell_count}):\n"
        
        if recommendations.get("sell_recommendations"):
            for sell in recommendations["sell_recommendations"]:
                body_text += f"- {sell['ticker']}: {sell['reason']}\n"
        else:
            body_text += "None\n"
        
        body_text += f"\nHOLD Positions: {hold_count} stocks\n"
        
        return {
            "subject": subject,
            "body_html": body_html,
            "body_text": body_text
        }
    
    def execute_auto_trades(self, recommendations: Dict, max_position_size: float = 5000) -> Dict:
        """Execute trades automatically based on recommendations.
        
        WARNING: This will execute real trades. Use with caution.
        
        Parameters
        ----------
        recommendations : dict
            Trading recommendations
        max_position_size : float
            Maximum position size per trade
            
        Returns
        -------
        dict
            Execution results
        """
        executed_trades = []
        
        # Get current portfolio state
        positions, cash = self.portfolio_analyzer.get_current_portfolio_state()
        
        # Execute BUY orders
        for candidate in recommendations.get("buy_candidates", []):
            if candidate.get("confidence") == "HIGH" and cash >= max_position_size:
                ticker = candidate["ticker"]
                price = candidate["current_price"]
                shares = int(max_position_size / price)
                
                if shares > 0:
                    # Would execute trade here - for safety, just log it
                    executed_trades.append({
                        "type": "BUY",
                        "ticker": ticker,
                        "shares": shares,
                        "price": price,
                        "total": shares * price,
                        "status": "simulated"  # Change to "executed" when ready
                    })
                    
                    # Update cash for next iteration
                    cash -= shares * price
        
        # Execute SELL orders
        for sell in recommendations.get("sell_recommendations", []):
            if sell.get("confidence") == "HIGH":
                ticker = sell["ticker"]
                
                # Find position
                for pos in positions:
                    if pos["ticker"] == ticker:
                        executed_trades.append({
                            "type": "SELL",
                            "ticker": ticker,
                            "shares": pos["shares"],
                            "status": "simulated"  # Change to "executed" when ready
                        })
                        break
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "executed_trades": executed_trades,
            "summary": {
                "total_trades": len(executed_trades),
                "buy_orders": len([t for t in executed_trades if t["type"] == "BUY"]),
                "sell_orders": len([t for t in executed_trades if t["type"] == "SELL"])
            }
        }


def main():
    """Command-line interface for n8n automation."""
    parser = argparse.ArgumentParser(description="N8N Trading Automation")
    parser.add_argument('command', choices=['analyze', 'status', 'execute'],
                       help='Command to execute')
    parser.add_argument('--tickers', nargs='+', 
                       help='Stock tickers to analyze (space-separated)')
    parser.add_argument('--watchlist-file', type=str,
                       help='Path to file containing ticker list (one per line)')
    parser.add_argument('--output', choices=['json', 'email'], default='json',
                       help='Output format')
    parser.add_argument('--auto-execute', action='store_true',
                       help='Automatically execute high-confidence trades (USE WITH CAUTION)')
    
    args = parser.parse_args()
    
    automation = N8NTradingAutomation()
    
    try:
        if args.command == 'analyze':
            # Get tickers from arguments or file
            tickers = []
            
            if args.tickers:
                tickers = [t.upper() for t in args.tickers]
            elif args.watchlist_file:
                with open(args.watchlist_file, 'r') as f:
                    tickers = [line.strip().upper() for line in f if line.strip()]
            else:
                # Default watchlist
                tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
            
            # Analyze tickers
            result = automation.analyze_watchlist(tickers)
            
            # Auto-execute if requested
            if args.auto_execute and result.get("status") == "success":
                execution_result = automation.execute_auto_trades(result["recommendations"])
                result["execution"] = execution_result
            
            # Output results
            if args.output == 'email':
                # Output email content only
                email_content = result.get("email_content", {})
                print(json.dumps(email_content, indent=2))
            else:
                print(json.dumps(result, indent=2))
                
        elif args.command == 'status':
            result = automation.get_portfolio_status()
            print(json.dumps(result, indent=2))
            
        elif args.command == 'execute':
            # Load recommendations from stdin
            recommendations = json.load(sys.stdin)
            result = automation.execute_auto_trades(recommendations)
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        error_result = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()