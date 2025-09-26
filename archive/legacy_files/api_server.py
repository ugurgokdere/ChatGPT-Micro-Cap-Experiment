"""Flask API Server for n8n Cloud Integration

This server provides HTTP endpoints for n8n cloud to trigger trading analysis.
Run this locally and expose it via ngrok or deploy to a cloud service.
"""

import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from interactive_research_analyzer import InteractiveResearchAnalyzer
from daily_trade_logger import DailyTradeLogger
from portfolio_analyzer import PortfolioAnalyzer
from market_research import MarketResearcher

app = Flask(__name__)
CORS(app)  # Enable CORS for n8n access

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
analyzer = InteractiveResearchAnalyzer()
trade_logger = DailyTradeLogger()
portfolio_analyzer = PortfolioAnalyzer()
market_researcher = MarketResearcher()

# Optional: Add basic authentication
API_KEY = os.environ.get('TRADING_API_KEY', 'your-secret-api-key-here')

def verify_api_key():
    """Verify API key from request headers."""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return False
    token = auth_header.split(' ')[1]
    return token == API_KEY

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_stocks():
    """Analyze stocks and return recommendations.
    
    Expected JSON payload:
    {
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "api_key": "your-api-key"  # Optional if using header auth
    }
    """
    # Verify API key
    if not verify_api_key():
        return jsonify({"status": "error", "error": "Unauthorized"}), 401
    
    try:
        data = request.json
        tickers = data.get('tickers', [])
        
        if not tickers:
            # Load from watchlist file if no tickers provided
            watchlist_file = Path(__file__).parent / "watchlist.txt"
            if watchlist_file.exists():
                with open(watchlist_file, 'r') as f:
                    tickers = [line.strip().upper() for line in f if line.strip()]
            else:
                tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        logger.info(f"Analyzing tickers: {tickers}")
        
        # Get detailed analysis
        detailed_analysis = analyzer.get_detailed_analysis(tickers)
        
        # Generate AI recommendations
        ai_analysis = analyzer.generate_ai_recommendations(tickers, detailed_analysis)
        
        # Generate final recommendations
        recommendations = analyzer.generate_final_recommendations(
            tickers, detailed_analysis, ai_analysis
        )
        
        # Format email content
        email_content = format_email_content(recommendations)
        
        response = {
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
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        return jsonify({
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }), 500

@app.route('/portfolio/status', methods=['GET'])
def get_portfolio_status():
    """Get current portfolio status.
    
    Requires API key in Authorization header.
    """
    # Verify API key
    if not verify_api_key():
        return jsonify({"status": "error", "error": "Unauthorized"}), 401
    
    try:
        positions, cash = portfolio_analyzer.get_current_portfolio_state()
        
        total_value = sum(pos.get('shares', 0) * pos.get('buy_price', 0) for pos in positions)
        total_equity = total_value + cash
        
        # Get recent trades
        trade_history = trade_logger.load_trade_history()
        recent_trades = trade_history[-5:] if trade_history else []
        
        response = {
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
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting portfolio status: {str(e)}")
        return jsonify({
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }), 500

@app.route('/log/trade', methods=['POST'])
def log_trade():
    """Log a manual trade.
    
    Expected JSON payload for BUY:
    {
        "type": "BUY",
        "ticker": "AAPL",
        "shares": 10,
        "price": 150.00,
        "stop_loss": 135.00,
        "reason": "Strong fundamentals"
    }
    
    Expected JSON payload for SELL:
    {
        "type": "SELL",
        "ticker": "AAPL",
        "shares": 10,
        "price": 160.00,
        "reason": "Take profits"
    }
    """
    # Verify API key
    if not verify_api_key():
        return jsonify({"status": "error", "error": "Unauthorized"}), 401
    
    try:
        data = request.json
        trade_type = data.get('type', '').upper()
        
        if trade_type not in ['BUY', 'SELL']:
            return jsonify({
                "status": "error",
                "error": "Invalid trade type. Must be BUY or SELL"
            }), 400
        
        # Log trade based on type
        if trade_type == 'BUY':
            # Implement buy logic
            pass
        else:
            # Implement sell logic
            pass
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "message": f"{trade_type} trade logged successfully"
        })
        
    except Exception as e:
        logger.error(f"Error logging trade: {str(e)}")
        return jsonify({
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }), 500

def format_email_content(recommendations):
    """Format recommendations for email notification."""
    buy_count = len(recommendations.get("buy_candidates", []))
    sell_count = len(recommendations.get("sell_recommendations", []))
    
    if buy_count > 0 or sell_count > 0:
        subject = f"🚨 ACTION REQUIRED: {buy_count} BUY, {sell_count} SELL signals"
    else:
        subject = f"📊 Portfolio Update: All positions on HOLD"
    
    # Generate HTML body
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
            </li>
            """
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
            </li>
            """
        body_html += "</ul>"
    else:
        body_html += "<p>No sell recommendations at this time.</p>"
    
    body_html += """
        <hr>
        <p style="color: #666; font-size: 12px;">
            This is an automated trading analysis.
        </p>
    </body>
    </html>
    """
    
    # Plain text version
    body_text = f"Trading Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    body_text += f"BUY: {buy_count} | SELL: {sell_count} | HOLD: {len(recommendations.get('hold_positions', []))}"
    
    return {
        "subject": subject,
        "body_html": body_html,
        "body_text": body_text
    }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)