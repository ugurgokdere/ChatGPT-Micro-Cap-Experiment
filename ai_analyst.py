"""AI-powered portfolio analyst using OpenAI/ChatGPT for trading decisions.

This module integrates with OpenAI's API to provide automated portfolio analysis
and trading recommendations based on the experiment's proven prompts.
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

try:
    from openai import OpenAI
except ImportError:
    print("OpenAI library not installed. Run: pip install openai")
    OpenAI = None


class AIPortfolioAnalyst:
    """AI-powered portfolio analyst for micro-cap trading decisions."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the AI analyst with OpenAI API key.
        
        Parameters
        ----------
        api_key : str, optional
            OpenAI API key. If not provided, will try to load from env.local or environment.
        """
        self.api_key = api_key or self._load_openai_key()
        
        # Initialize OpenAI client
        if OpenAI and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
        
        # Experiment constraints
        self.max_market_cap = 300_000_000  # $300M micro-cap limit
        self.experiment_start = datetime(2025, 6, 27)
        self.experiment_end = datetime(2025, 12, 27)
        
    def _load_openai_key(self) -> str:
        """Load OpenAI API key from env.local file or environment variables."""
        # Try environment variable first
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return api_key
            
        # Try env.local file
        env_file = Path(__file__).resolve().parent / "env.local"
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY="):
                        return line.split("=", 1)[1]
        
        print("Warning: No OpenAI API key found. Set OPENAI_API_KEY in env.local or environment.")
        return ""
    
    def get_portfolio_analysis_prompt(self, portfolio_data: Dict, cash_balance: float, 
                                    current_performance: Dict, last_thesis: str = "") -> str:
        """Generate comprehensive portfolio analysis prompt based on experiment prompts.
        
        Parameters
        ----------
        portfolio_data : dict
            Current portfolio positions
        cash_balance : float
            Available cash balance
        current_performance : dict
            Performance metrics vs benchmarks
        last_thesis : str, optional
            Previous investment thesis
            
        Returns
        -------
        str
            Formatted prompt for AI analysis
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        days_remaining = (self.experiment_end - datetime.now()).days
        
        prompt = f"""You are a professional-grade portfolio strategist managing a micro-cap experiment.

CURRENT SITUATION:
- Date: {current_date}
- Days remaining in experiment: {days_remaining} (ends 2025-12-27)
- Available cash: ${cash_balance:.2f}
- Current portfolio: {portfolio_data}

PERFORMANCE METRICS:
- Portfolio value: ${current_performance.get('total_equity', 0):.2f}
- Return vs S&P 500: {current_performance.get('vs_sp500', 'N/A')}
- Total return: {current_performance.get('total_return', 'N/A')}

CONSTRAINTS:
- ONLY U.S.-listed micro-cap stocks (market cap under $300M)
- Full-share positions only
- Must maximize return from start (2025-06-27) to end (2025-12-27)
- Complete control over position sizing, risk management, stop-losses

PREVIOUS THESIS:
{last_thesis or "No previous thesis available."}

ANALYSIS REQUIRED:
1. Evaluate current holdings for micro-cap compliance
2. Assess if any positions should be sold (risk/reward analysis)
3. Research potential new micro-cap opportunities
4. Consider market conditions and remaining time horizon
5. Recommend specific buy/sell actions with reasoning

RESPONSE FORMAT:
Please structure your response as:
1. CURRENT PORTFOLIO ASSESSMENT
2. MARKET OUTLOOK
3. RECOMMENDED ACTIONS
4. NEW INVESTMENT THESIS (short summary)

Focus on actionable decisions that maximize alpha generation within the micro-cap universe."""

        return prompt
    
    def get_deep_research_prompt(self, portfolio_data: Dict, cash_balance: float, 
                               last_thesis: str = "") -> str:
        """Generate deep research prompt for weekly portfolio reevaluation.
        
        Based on the experiment's "All deep research prompts going forward" template.
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""You are a professional grade portfolio analyst. Use deep research to reevaluate your portfolio.

CURRENT PORTFOLIO ({current_date}):
{portfolio_data}

AVAILABLE CASH: ${cash_balance:.2f}

LAST THESIS:
{last_thesis or "No previous thesis available."}

INSTRUCTIONS:
- Check current holdings and find new stocks
- Complete control as long as it is a micro cap (market cap under $300M)
- Buy anything as long as you have the capital available
- Remember your only goal is alpha

DEEP RESEARCH AREAS:
1. Current holdings fundamental analysis
2. Micro-cap market screening for new opportunities
3. Risk assessment and position sizing
4. Market timing and catalysts
5. Competitive positioning analysis

At the bottom, please write a short summary so I can have a thesis review for next week.

Use comprehensive research to make data-driven decisions that maximize portfolio returns."""

        return prompt
    
    def analyze_portfolio(self, portfolio_data: Dict, cash_balance: float, 
                         current_performance: Dict, last_thesis: str = "") -> Dict:
        """Get AI-powered portfolio analysis and trading recommendations.
        
        Parameters
        ----------
        portfolio_data : dict
            Current portfolio positions
        cash_balance : float
            Available cash balance  
        current_performance : dict
            Performance metrics
        last_thesis : str, optional
            Previous investment thesis
            
        Returns
        -------
        dict
            AI analysis results with recommendations
        """
        if not self.client:
            return {
                "error": "OpenAI client not available. Check API key configuration.",
                "recommendations": [],
                "thesis": ""
            }
        
        prompt = self.get_portfolio_analysis_prompt(
            portfolio_data, cash_balance, current_performance, last_thesis
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert portfolio strategist specializing in micro-cap stocks."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            analysis_text = response.choices[0].message.content
            
            # Parse the response for actionable recommendations
            recommendations = self._parse_recommendations(analysis_text)
            thesis = self._extract_thesis(analysis_text)
            
            return {
                "analysis": analysis_text,
                "recommendations": recommendations,
                "thesis": thesis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "error": f"OpenAI API error: {str(e)}",
                "recommendations": [],
                "thesis": ""
            }
    
    def _parse_recommendations(self, analysis_text: str) -> List[Dict]:
        """Parse AI response to extract actionable trading recommendations."""
        recommendations = []
        
        # Look for common trading action keywords
        lines = analysis_text.split('\n')
        current_action = None
        
        for line in lines:
            line = line.strip().lower()
            
            # Detect buy recommendations
            if any(word in line for word in ['buy', 'purchase', 'acquire', 'add']):
                if any(word in line for word in ['$', 'shares', 'position']):
                    recommendations.append({
                        "action": "BUY",
                        "text": line,
                        "confidence": "medium"
                    })
            
            # Detect sell recommendations  
            elif any(word in line for word in ['sell', 'exit', 'close', 'dump']):
                if any(word in line for word in ['position', 'shares', 'holding']):
                    recommendations.append({
                        "action": "SELL", 
                        "text": line,
                        "confidence": "medium"
                    })
        
        return recommendations
    
    def _extract_thesis(self, analysis_text: str) -> str:
        """Extract investment thesis from AI response."""
        # Look for thesis section
        lines = analysis_text.split('\n')
        thesis_started = False
        thesis_lines = []
        
        for line in lines:
            if any(word in line.lower() for word in ['thesis', 'summary', 'strategy']):
                thesis_started = True
                continue
                
            if thesis_started:
                if line.strip():
                    thesis_lines.append(line.strip())
                    
                # Stop at next major section
                if line.startswith('#') or line.isupper():
                    break
        
        return ' '.join(thesis_lines[:3])  # First 3 relevant lines
    
    def validate_micro_cap_compliance(self, ticker: str, market_cap: float) -> bool:
        """Validate if a stock meets micro-cap criteria.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        market_cap : float
            Market capitalization in dollars
            
        Returns
        -------
        bool
            True if stock qualifies as micro-cap
        """
        return market_cap < self.max_market_cap
    
    def get_experiment_status(self) -> Dict:
        """Get current experiment status and timeline."""
        now = datetime.now()
        days_elapsed = (now - self.experiment_start).days
        days_remaining = (self.experiment_end - now).days
        total_days = (self.experiment_end - self.experiment_start).days
        
        return {
            "start_date": self.experiment_start.strftime("%Y-%m-%d"),
            "end_date": self.experiment_end.strftime("%Y-%m-%d"),
            "current_date": now.strftime("%Y-%m-%d"),
            "days_elapsed": days_elapsed,
            "days_remaining": days_remaining,
            "total_days": total_days,
            "progress_pct": (days_elapsed / total_days) * 100
        }


def load_current_portfolio() -> Tuple[Dict, float]:
    """Load current portfolio state from CSV files."""
    from trading_script import load_latest_portfolio_state, PORTFOLIO_CSV
    
    try:
        portfolio_data, cash_balance = load_latest_portfolio_state(str(PORTFOLIO_CSV))
        return portfolio_data, cash_balance
    except Exception as e:
        print(f"Error loading portfolio: {e}")
        return {}, 100.0


if __name__ == "__main__":
    # Example usage
    analyst = AIPortfolioAnalyst()
    
    # Load current portfolio
    portfolio, cash = load_current_portfolio()
    
    # Mock performance data
    performance = {
        "total_equity": 12687.44,
        "vs_sp500": "+15.2%", 
        "total_return": "+26.87%"
    }
    
    # Get AI analysis
    result = analyst.analyze_portfolio(portfolio, cash, performance)
    
    if "error" not in result:
        print("=== AI Portfolio Analysis ===")
        print(result["analysis"])
        print(f"\nRecommendations: {len(result['recommendations'])}")
        for rec in result["recommendations"]:
            print(f"- {rec['action']}: {rec['text']}")
    else:
        print(f"Error: {result['error']}")