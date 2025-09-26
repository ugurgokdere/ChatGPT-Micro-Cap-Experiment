"""Interactive Weekly Research Analyzer

This script allows manual input of stocks from weekly research reports
and provides AI-powered analysis and buy/sell recommendations.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ai_analyst import AIPortfolioAnalyst  
from market_research import MarketResearcher
from portfolio_analyzer import PortfolioAnalyzer
from trading_script import set_data_dir


class InteractiveResearchAnalyzer:
    """Interactive analyzer for manually inputted research stocks."""
    
    def __init__(self, openai_api_key: Optional[str] = None, finnhub_api_key: Optional[str] = None):
        """Initialize interactive research analyzer."""
        self.ai_analyst = AIPortfolioAnalyst(openai_api_key)
        self.market_researcher = MarketResearcher(finnhub_api_key)
        self.portfolio_analyzer = PortfolioAnalyzer(finnhub_api_key)
        
        # Set data directory to Start Your Own
        start_your_own_dir = Path(__file__).parent / "Start Your Own"
        set_data_dir(start_your_own_dir)
        
    def collect_research_stocks(self) -> List[str]:
        """Interactive collection of stock tickers from user."""
        print("📝 WEEKLY RESEARCH STOCK COLLECTION")
        print("=" * 50)
        print("Enter the stock tickers from your weekly research report.")
        print("Type 'done' when finished, or 'quit' to exit.\n")
        
        stocks = []
        
        while True:
            ticker_input = input(f"Enter stock ticker #{len(stocks) + 1} (or 'done'/'quit'): ").strip().upper()
            
            if ticker_input in ['DONE', 'D', '']:
                if stocks:
                    break
                else:
                    print("Please enter at least one stock ticker.")
                    continue
                    
            if ticker_input in ['QUIT', 'Q', 'EXIT']:
                print("❌ Exiting...")
                return []
            
            # Validate ticker format
            if not ticker_input or not ticker_input.isalpha() or len(ticker_input) > 5:
                print("❌ Invalid ticker format. Please enter 1-5 letters only.")
                continue
                
            # Check if already added
            if ticker_input in stocks:
                print(f"⚠️ {ticker_input} already added.")
                continue
                
            # Try to validate the ticker exists
            try:
                print(f"🔍 Validating {ticker_input}...")
                fundamentals = self.market_researcher.get_stock_fundamentals(ticker_input)
                
                if fundamentals.get('error') or fundamentals.get('current_price', 0) <= 0:
                    print(f"❌ {ticker_input}: Could not find valid stock data.")
                    retry = input("Add anyway? (y/N): ").lower().strip()
                    if retry not in ['y', 'yes']:
                        continue
                        
                current_price = fundamentals.get('current_price', 0)
                company_name = fundamentals.get('company_name', 'Unknown')
                print(f"✅ {ticker_input}: {company_name} - ${current_price:.2f}")
                
            except Exception as e:
                print(f"⚠️ Error validating {ticker_input}: {e}")
                retry = input("Add anyway? (y/N): ").lower().strip()
                if retry not in ['y', 'yes']:
                    continue
            
            stocks.append(ticker_input)
            print(f"📊 Added {ticker_input} to research list.\n")
        
        print(f"\n✅ Collected {len(stocks)} stocks: {', '.join(stocks)}")
        return stocks
    
    def get_detailed_analysis(self, tickers: List[str]) -> Dict:
        """Get detailed analysis for each stock."""
        detailed_analysis = {}
        
        print("\n📊 DETAILED STOCK ANALYSIS")
        print("=" * 40)
        
        for ticker in tickers:
            print(f"\n🔍 Analyzing {ticker}...")
            
            try:
                # Get comprehensive research
                research_report = self.market_researcher.research_stock(ticker)
                
                # Extract key metrics
                fundamentals = research_report.get('fundamentals', {})
                assessment = research_report.get('assessment', {})
                
                detailed_analysis[ticker] = {
                    "ticker": ticker,
                    "current_price": fundamentals.get('current_price', 0),
                    "market_cap": fundamentals.get('market_cap', 0),
                    "is_micro_cap": fundamentals.get('is_micro_cap', False),
                    "micro_cap_eligible": fundamentals.get('micro_cap_eligible', False),
                    "company_name": fundamentals.get('company_name', ''),
                    "industry": fundamentals.get('industry', ''),
                    "pe_ratio": fundamentals.get('pe_ratio'),
                    "revenue_growth": fundamentals.get('revenue_growth'),
                    "assessment_score": assessment.get('score', 0),
                    "recommendation": assessment.get('recommendation', 'HOLD'),
                    "key_factors": assessment.get('factors', []),
                    "compliance_issues": []
                }
                
                # Check compliance - now allowing all cap sizes
                # Note: Removed micro-cap requirement to allow all market cap sizes
                
                if fundamentals.get("country") != "US":
                    detailed_analysis[ticker]["compliance_issues"].append("Not US-listed")
                
                if fundamentals.get('current_price', 0) < 1.0:
                    detailed_analysis[ticker]["compliance_issues"].append("Penny stock")
                
                # Print summary
                analysis = detailed_analysis[ticker]
                print(f"   Company: {analysis['company_name']}")
                print(f"   Price: ${analysis['current_price']:.2f}")
                print(f"   Market Cap: ${analysis['market_cap']:,.0f}")
                print(f"   Micro-cap: {'✅' if analysis['is_micro_cap'] else '❌'}")
                print(f"   Assessment Score: {analysis['assessment_score']}/7")
                
                if analysis['compliance_issues']:
                    print(f"   ⚠️ Issues: {', '.join(analysis['compliance_issues'])}")
                    
            except Exception as e:
                print(f"❌ Error analyzing {ticker}: {e}")
                detailed_analysis[ticker] = {
                    "ticker": ticker,
                    "error": str(e)
                }
        
        return detailed_analysis
    
    def generate_ai_recommendations(self, tickers: List[str], detailed_analysis: Dict) -> Dict:
        """Generate AI-powered buy/sell recommendations."""
        print("\n🤖 AI ANALYSIS & RECOMMENDATIONS")
        print("=" * 40)
        
        # Get current portfolio state
        positions, cash = self.portfolio_analyzer.get_current_portfolio_state()
        current_tickers = [pos['ticker'] for pos in positions] if positions else []
        
        # Create research summary for AI
        research_summary = "Weekly Research Stocks Analysis:\n\n"
        for ticker in tickers:
            analysis = detailed_analysis.get(ticker, {})
            if 'error' not in analysis:
                research_summary += f"{ticker} ({analysis.get('company_name', 'Unknown')}):\n"
                research_summary += f"- Price: ${analysis.get('current_price', 0):.2f}\n"
                research_summary += f"- Market Cap: ${analysis.get('market_cap', 0):,.0f}\n"
                research_summary += f"- Industry: {analysis.get('industry', 'Unknown')}\n"
                research_summary += f"- Micro-cap: {analysis.get('is_micro_cap', False)}\n"
                research_summary += f"- Assessment Score: {analysis.get('assessment_score', 0)}/7\n"
                if analysis.get('key_factors'):
                    research_summary += f"- Key Factors: {', '.join(analysis.get('key_factors', [])[:3])}\n"
                research_summary += "\n"
        
        # Create analysis prompt
        prompt = f"""You are analyzing weekly research stocks for trading decisions across all market cap sizes.

CURRENT PORTFOLIO:
- Current positions: {current_tickers if current_tickers else 'None (cash only)'}
- Available cash: ${cash:.2f}
- Total equity: ${sum(pos.get('shares', 0) * pos.get('buy_price', 0) for pos in positions) + cash:.2f}

RESEARCH STOCKS TO ANALYZE:
{research_summary}

ANALYSIS REQUIRED:
For each stock, provide specific BUY/SELL/HOLD recommendations considering:

1. Market capitalization and liquidity (all cap sizes now accepted)
2. Financial strength and growth potential
3. Current portfolio diversification
4. Risk/reward ratio
5. Upcoming catalysts or concerns

For each recommendation, specify:
- Action: BUY/SELL/HOLD
- Confidence: HIGH/MEDIUM/LOW
- Reasoning: Key factors driving the decision
- Position size: Suggested dollar amount if buying
- Risk factors to monitor

Focus on actionable recommendations for the next 1-3 months."""

        try:
            # Get AI analysis
            performance_data = {
                'total_equity': sum(pos.get('shares', 0) * pos.get('buy_price', 0) for pos in positions) + cash,
                'vs_sp500': 'N/A',
                'total_return': 'N/A'
            }
            
            ai_result = self.ai_analyst.analyze_portfolio(
                {ticker: {"current_research": "weekly research candidate"} for ticker in tickers},
                cash,
                performance_data,
                prompt
            )
            
            return {
                "research_tickers": tickers,
                "current_portfolio": current_tickers,
                "cash_available": cash,
                "ai_analysis": ai_result,
                "analysis_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"AI analysis failed: {e}"}
    
    def generate_final_recommendations(self, tickers: List[str], detailed_analysis: Dict, ai_analysis: Dict) -> Dict:
        """Generate final structured recommendations."""
        recommendations = {
            "analysis_date": datetime.now().isoformat(),
            "research_source": "Manual input",
            "stocks_analyzed": tickers,
            "detailed_analysis": detailed_analysis,
            "ai_analysis": ai_analysis,
            "buy_candidates": [],
            "sell_recommendations": [],
            "hold_positions": []
        }
        
        # Process each stock
        for ticker in tickers:
            analysis = detailed_analysis.get(ticker, {})
            
            if 'error' in analysis:
                continue
                
            # Determine recommendation based on multiple factors
            is_micro_cap = analysis.get('is_micro_cap', False)
            assessment_score = analysis.get('assessment_score', 0)
            compliance_issues = analysis.get('compliance_issues', [])
            current_price = analysis.get('current_price', 0)
            
            # High conviction buys: high score + no major issues (any cap size)
            if assessment_score >= 4 and not compliance_issues:
                cap_type = "micro-cap" if is_micro_cap else "larger-cap"
                recommendations["buy_candidates"].append({
                    "ticker": ticker,
                    "current_price": current_price,
                    "reason": f"High-quality {cap_type} stock with score {assessment_score}/7",
                    "confidence": "HIGH" if assessment_score >= 5 else "MEDIUM",
                    "assessment_score": assessment_score,
                    "key_factors": analysis.get('key_factors', [])[:3]
                })
            # Moderate opportunities (any cap size)
            elif assessment_score >= 2:
                cap_type = "micro-cap" if is_micro_cap else "larger-cap"
                recommendations["hold_positions"].append({
                    "ticker": ticker,
                    "current_price": current_price,
                    "reason": f"{cap_type.title()} with moderate potential (score: {assessment_score}/7)",
                    "assessment_score": assessment_score
                })
            # Non-compliant or low-quality
            else:
                issue_summary = ', '.join(compliance_issues) if compliance_issues else "Low assessment score"
                recommendations["hold_positions"].append({
                    "ticker": ticker,
                    "current_price": current_price,
                    "reason": f"Monitor only - {issue_summary}",
                    "assessment_score": assessment_score
                })
        
        # Check current portfolio for sells (if any positions not in research)
        positions, _ = self.portfolio_analyzer.get_current_portfolio_state()
        for pos in positions:
            ticker = pos['ticker']
            if ticker not in tickers:
                recommendations["sell_recommendations"].append({
                    "ticker": ticker,
                    "reason": "Not in current week's research focus",
                    "current_value": pos.get('shares', 0) * pos.get('buy_price', 0),
                    "confidence": "MEDIUM"
                })
        
        return recommendations
    
    def print_recommendations_summary(self, recommendations: Dict):
        """Print formatted summary of recommendations."""
        print("\n" + "=" * 60)
        print("📊 FINAL WEEKLY RECOMMENDATIONS")
        print("=" * 60)
        
        # Analysis summary
        stocks_analyzed = recommendations.get('stocks_analyzed', [])
        print(f"📈 Stocks Analyzed: {len(stocks_analyzed)}")
        if stocks_analyzed:
            print(f"   {', '.join(stocks_analyzed)}")
        
        # Buy candidates
        buy_candidates = recommendations.get('buy_candidates', [])
        print(f"\n💰 BUY CANDIDATES: {len(buy_candidates)}")
        if buy_candidates:
            for candidate in buy_candidates:
                print(f"   🔹 {candidate['ticker']} @ ${candidate['current_price']:.2f}")
                print(f"      Confidence: {candidate['confidence']} (Score: {candidate['assessment_score']}/7)")
                print(f"      Reason: {candidate['reason']}")
                if candidate.get('key_factors'):
                    print(f"      Key Factors: {', '.join(candidate['key_factors'])}")
                print()
        else:
            print("   No strong buy candidates identified.")
        
        # Sell recommendations  
        sell_recommendations = recommendations.get('sell_recommendations', [])
        print(f"📉 SELL RECOMMENDATIONS: {len(sell_recommendations)}")
        if sell_recommendations:
            for sell in sell_recommendations:
                print(f"   🔸 {sell['ticker']}")
                print(f"      Reason: {sell['reason']}")
                print(f"      Confidence: {sell['confidence']}")
        else:
            print("   No sell recommendations.")
        
        # Hold/Monitor positions
        hold_positions = recommendations.get('hold_positions', [])
        print(f"\n📊 HOLD/MONITOR: {len(hold_positions)}")
        if hold_positions:
            for hold in hold_positions:
                print(f"   ⚪ {hold['ticker']} @ ${hold['current_price']:.2f}")
                print(f"      Reason: {hold['reason']}")
        
        # AI Analysis Summary
        ai_analysis = recommendations.get('ai_analysis', {})
        if ai_analysis and not ai_analysis.get('error'):
            print(f"\n🤖 AI ANALYSIS SUMMARY:")
            analysis_text = ai_analysis.get('analysis', '')
            if analysis_text:
                # Print first few lines of AI analysis
                lines = analysis_text.split('\n')[:5]
                for line in lines:
                    if line.strip():
                        print(f"   {line.strip()}")
                if len(analysis_text.split('\n')) > 5:
                    print("   [... see full analysis in saved file]")
        
        print(f"\n✨ Analysis completed at {datetime.now().strftime('%H:%M:%S')}")
    
    def save_recommendations(self, recommendations: Dict, filename: Optional[str] = None):
        """Save recommendations to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"interactive_analysis_{timestamp}.json"
        
        output_dir = Path(__file__).parent / "weekly_analysis"
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(recommendations, f, indent=2, default=str)
            
            print(f"💾 Full analysis saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving analysis: {e}")
            return None
    
    def run_interactive_session(self):
        """Run the complete interactive research analysis session."""
        print("🔬 INTERACTIVE WEEKLY RESEARCH ANALYZER")
        print("=" * 60)
        print("Analyze stocks from your weekly research with AI recommendations.\n")
        
        try:
            # Step 1: Collect stocks
            stocks = self.collect_research_stocks()
            if not stocks:
                return
                
            # Step 2: Get detailed analysis
            detailed_analysis = self.get_detailed_analysis(stocks)
            
            # Step 3: AI recommendations
            ai_analysis = self.generate_ai_recommendations(stocks, detailed_analysis)
            
            # Step 4: Generate final recommendations
            recommendations = self.generate_final_recommendations(stocks, detailed_analysis, ai_analysis)
            
            # Step 5: Display results
            self.print_recommendations_summary(recommendations)
            
            # Step 6: Save results
            self.save_recommendations(recommendations)
            
            print(f"\n✅ Analysis complete! Analyzed {len(stocks)} stocks with AI recommendations.")
            
        except KeyboardInterrupt:
            print("\n\n❌ Analysis interrupted by user.")
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")


def main():
    """Command-line interface for interactive research analyzer."""
    try:
        analyzer = InteractiveResearchAnalyzer()
        analyzer.run_interactive_session()
    except Exception as e:
        print(f"❌ Analyzer failed to start: {e}")


if __name__ == "__main__":
    main()