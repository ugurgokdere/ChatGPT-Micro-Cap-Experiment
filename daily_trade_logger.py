"""Interactive Daily Trade Logger for syncing actual trading decisions.

This module provides an interactive interface to log actual trades made
after receiving AI recommendations, keeping the system in sync with reality.
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from trading_script import log_manual_buy, log_manual_sell, load_latest_portfolio_state, set_data_dir
from portfolio_analyzer import PortfolioAnalyzer
from market_research import MarketResearcher


class DailyTradeLogger:
    """Interactive system for logging actual daily trades."""
    
    def __init__(self, finnhub_api_key: Optional[str] = None):
        """Initialize daily trade logger.
        
        Parameters
        ----------
        finnhub_api_key : str, optional
            Finnhub API key for price validation
        """
        self.market_researcher = MarketResearcher(finnhub_api_key)
        self.portfolio_analyzer = PortfolioAnalyzer(finnhub_api_key)
        
        # Set data directory to Start Your Own
        start_your_own_dir = Path(__file__).parent / "Start Your Own"
        set_data_dir(start_your_own_dir)
        
        # Trade log file
        self.trade_log_file = Path(__file__).parent / "daily_trade_decisions.json"
        
    def load_trade_history(self) -> List[Dict]:
        """Load previous trade decisions from log file.
        
        Returns
        -------
        list
            List of previous trade decisions
        """
        if not self.trade_log_file.exists():
            return []
        
        try:
            with open(self.trade_log_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading trade history: {e}")
            return []
    
    def save_trade_history(self, trades: List[Dict]):
        """Save trade decisions to log file.
        
        Parameters
        ----------
        trades : list
            List of trade decisions to save
        """
        try:
            with open(self.trade_log_file, 'w') as f:
                json.dump(trades, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving trade history: {e}")
    
    def get_current_portfolio_summary(self) -> Dict:
        """Get current portfolio state summary.
        
        Returns
        -------
        dict
            Current portfolio summary
        """
        positions, cash = self.portfolio_analyzer.get_current_portfolio_state()
        
        total_value = sum(pos.get('shares', 0) * pos.get('buy_price', 0) for pos in positions)
        
        return {
            "positions": len(positions),
            "total_invested": total_value,
            "cash_available": cash,
            "total_equity": total_value + cash,
            "holdings": [
                {
                    "ticker": pos['ticker'],
                    "shares": pos['shares'],
                    "cost_basis": pos['buy_price'],
                    "current_value": pos['shares'] * pos['buy_price']
                }
                for pos in positions
            ]
        }
    
    def validate_trade_price(self, ticker: str, price: float, trade_type: str) -> bool:
        """Validate that trade price is reasonable for current market.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        price : float
            Proposed trade price
        trade_type : str
            'BUY' or 'SELL'
            
        Returns
        -------
        bool
            True if price appears valid
        """
        try:
            # Get current market data
            fundamentals = self.market_researcher.get_stock_fundamentals(ticker)
            current_price = fundamentals.get('current_price', 0)
            
            if current_price == 0:
                print(f"⚠️ Warning: Could not validate current price for {ticker}")
                return True  # Allow trade but warn
            
            # Check if price is within reasonable range (±10% of current price)
            price_diff = abs(price - current_price) / current_price
            
            if price_diff > 0.10:  # More than 10% difference
                print(f"⚠️ Warning: {ticker} trade price ${price:.2f} differs from current ${current_price:.2f} by {price_diff*100:.1f}%")
                confirm = input("Continue anyway? (y/N): ").lower().strip()
                return confirm in ['y', 'yes']
            
            print(f"✅ {ticker} price ${price:.2f} validated (current: ${current_price:.2f})")
            return True
            
        except Exception as e:
            print(f"⚠️ Warning: Could not validate price for {ticker}: {e}")
            return True  # Allow trade but warn
    
    def log_buy_trade(self) -> bool:
        """Interactive buy trade logging.
        
        Returns
        -------
        bool
            True if trade was logged successfully
        """
        print("\n📈 LOGGING BUY TRADE")
        print("-" * 30)
        
        try:
            # Get trade details
            ticker = input("Enter ticker symbol: ").strip().upper()
            if not ticker:
                print("❌ Ticker required")
                return False
            
            shares_input = input("Enter number of shares: ").strip()
            try:
                shares = float(shares_input)
                if shares <= 0:
                    print("❌ Shares must be positive")
                    return False
            except ValueError:
                print("❌ Invalid shares amount")
                return False
            
            price_input = input("Enter buy price per share: ").strip()
            try:
                buy_price = float(price_input)
                if buy_price <= 0:
                    print("❌ Price must be positive")
                    return False
            except ValueError:
                print("❌ Invalid price")
                return False
            
            stop_loss_input = input("Enter stop loss price (optional): ").strip()
            try:
                stop_loss = float(stop_loss_input) if stop_loss_input else buy_price * 0.85  # 15% stop loss default
            except ValueError:
                print("❌ Invalid stop loss price")
                return False
            
            # Validate price
            if not self.validate_trade_price(ticker, buy_price, 'BUY'):
                return False
            
            # Get reason
            reason = input("Enter reason for purchase (optional): ").strip()
            if not reason:
                reason = "Manual trade entry"
            
            # Show trade summary
            total_cost = shares * buy_price
            print(f"\n📋 TRADE SUMMARY:")
            print(f"   Ticker: {ticker}")
            print(f"   Shares: {shares}")
            print(f"   Price: ${buy_price:.2f}")
            print(f"   Stop Loss: ${stop_loss:.2f}")
            print(f"   Total Cost: ${total_cost:.2f}")
            print(f"   Reason: {reason}")
            
            confirm = input("\nConfirm this trade? (y/N): ").lower().strip()
            if confirm not in ['y', 'yes']:
                print("❌ Trade cancelled")
                return False
            
            # Get current portfolio state
            positions, cash = self.portfolio_analyzer.get_current_portfolio_state()
            
            # Check if enough cash
            if total_cost > cash:
                print(f"❌ Insufficient cash. Need ${total_cost:.2f}, have ${cash:.2f}")
                return False
            
            # Log the trade using existing system
            try:
                # Create a DataFrame with proper structure for log_manual_buy
                if positions:
                    portfolio_df = pd.DataFrame(positions)
                else:
                    # Create empty DataFrame with correct columns
                    portfolio_df = pd.DataFrame(columns=['ticker', 'shares', 'buy_price', 'stop_loss', 'cost_basis'])
                
                new_cash, updated_portfolio = log_manual_buy(
                    buy_price=buy_price,
                    shares=shares,
                    ticker=ticker,
                    stoploss=stop_loss,
                    cash=cash,
                    chatgpt_portfolio=portfolio_df,
                    interactive=False  # Skip confirmation since we already confirmed
                )
                
                # Update the portfolio CSV file with the new position  
                from datetime import datetime
                
                # Use the correct path for the Start Your Own directory
                portfolio_csv_path = Path(__file__).parent / "Start Your Own" / "chatgpt_portfolio_update.csv"
                
                print("Updating portfolio CSV...")
                
                # Create the results for CSV update
                today = datetime.now().strftime("%Y-%m-%d")
                results = []
                
                # Add the new position row
                results.append({
                    "Date": today,
                    "Ticker": ticker,
                    "Shares": shares,
                    "Cost Basis": buy_price,
                    "Stop Loss": stop_loss,
                    "Current Price": buy_price,  # Use buy price as current for new positions
                    "Total Value": shares * buy_price,
                    "PnL": 0.0,
                    "Action": "BUY",
                    "Cash Balance": "",
                    "Total Equity": ""
                })
                
                # Add existing positions from updated_portfolio
                for _, stock in updated_portfolio.iterrows():
                    if stock["ticker"] != ticker:  # Don't duplicate the new position
                        stock_value = stock["shares"] * stock["buy_price"]
                        results.append({
                            "Date": today,
                            "Ticker": stock["ticker"],
                            "Shares": stock["shares"],
                            "Cost Basis": stock["buy_price"],
                            "Stop Loss": stock["stop_loss"],
                            "Current Price": stock["buy_price"],
                            "Total Value": stock_value,
                            "PnL": 0.0,
                            "Action": "",
                            "Cash Balance": "",
                            "Total Equity": ""
                        })
                
                # Calculate totals
                total_value = sum(row["Total Value"] for row in results)
                
                # Add TOTAL row
                results.append({
                    "Date": today,
                    "Ticker": "TOTAL",
                    "Shares": "",
                    "Cost Basis": "",
                    "Stop Loss": "",
                    "Current Price": "",
                    "Total Value": round(total_value, 2),
                    "PnL": 0.0,
                    "Action": "",
                    "Cash Balance": round(new_cash, 2),
                    "Total Equity": round(total_value + new_cash, 2),
                })
                
                # Save to CSV
                df = pd.DataFrame(results)
                if portfolio_csv_path.exists():
                    existing = pd.read_csv(portfolio_csv_path)
                    existing = existing[existing["Date"] != today]
                    df = pd.concat([existing, df], ignore_index=True)
                
                df.to_csv(portfolio_csv_path, index=False)
                
                # Log decision
                trade_record = {
                    "date": datetime.now().isoformat(),
                    "type": "BUY",
                    "ticker": ticker,
                    "shares": shares,
                    "price": buy_price,
                    "stop_loss": stop_loss,
                    "total_cost": total_cost,
                    "reason": reason,
                    "cash_before": cash,
                    "cash_after": new_cash
                }
                
                # Save to trade history
                trade_history = self.load_trade_history()
                trade_history.append(trade_record)
                self.save_trade_history(trade_history)
                
                print(f"✅ Buy trade logged successfully!")
                print(f"   New cash balance: ${new_cash:.2f}")
                
                return True
                
            except Exception as e:
                print(f"❌ Error logging trade: {e}")
                return False
                
        except KeyboardInterrupt:
            print("\n❌ Trade logging cancelled")
            return False
    
    def log_sell_trade(self) -> bool:
        """Interactive sell trade logging.
        
        Returns
        -------
        bool
            True if trade was logged successfully
        """
        print("\n📉 LOGGING SELL TRADE")
        print("-" * 30)
        
        # Show current positions
        positions, cash = self.portfolio_analyzer.get_current_portfolio_state()
        
        if not positions:
            print("❌ No positions to sell")
            return False
        
        print("Current positions:")
        for i, pos in enumerate(positions, 1):
            print(f"   {i}. {pos['ticker']}: {pos['shares']} shares @ ${pos['buy_price']:.2f}")
        
        try:
            # Get trade details
            ticker = input("\nEnter ticker symbol to sell: ").strip().upper()
            if not ticker:
                print("❌ Ticker required")
                return False
            
            # Check if we own this stock
            position = None
            for pos in positions:
                if pos['ticker'] == ticker:
                    position = pos
                    break
            
            if not position:
                print(f"❌ You don't own {ticker}")
                return False
            
            max_shares = position['shares']
            print(f"   You own {max_shares} shares of {ticker}")
            
            shares_input = input(f"Enter shares to sell (max {max_shares}): ").strip()
            try:
                shares = float(shares_input)
                if shares <= 0 or shares > max_shares:
                    print(f"❌ Invalid shares amount. Must be between 0 and {max_shares}")
                    return False
            except ValueError:
                print("❌ Invalid shares amount")
                return False
            
            price_input = input("Enter sell price per share: ").strip()
            try:
                sell_price = float(price_input)
                if sell_price <= 0:
                    print("❌ Price must be positive")
                    return False
            except ValueError:
                print("❌ Invalid price")
                return False
            
            # Validate price
            if not self.validate_trade_price(ticker, sell_price, 'SELL'):
                return False
            
            # Get reason
            reason = input("Enter reason for sale: ").strip()
            if not reason:
                reason = "Manual trade entry"
            
            # Calculate P&L
            cost_basis = position['buy_price']
            total_proceeds = shares * sell_price
            total_cost = shares * cost_basis
            pnl = total_proceeds - total_cost
            pnl_pct = (pnl / total_cost) * 100 if total_cost > 0 else 0
            
            # Show trade summary
            print(f"\n📋 TRADE SUMMARY:")
            print(f"   Ticker: {ticker}")
            print(f"   Shares: {shares}")
            print(f"   Sell Price: ${sell_price:.2f}")
            print(f"   Cost Basis: ${cost_basis:.2f}")
            print(f"   Total Proceeds: ${total_proceeds:.2f}")
            print(f"   P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
            print(f"   Reason: {reason}")
            
            confirm = input("\nConfirm this trade? (y/N): ").lower().strip()
            if confirm not in ['y', 'yes']:
                print("❌ Trade cancelled")
                return False
            
            # Log the trade using existing system
            try:
                # Create a simple DataFrame for the portfolio
                portfolio_df = pd.DataFrame(positions)
                
                new_cash, updated_portfolio = log_manual_sell(
                    sell_price=sell_price,
                    shares_sold=shares,
                    ticker=ticker,
                    cash=cash,
                    chatgpt_portfolio=portfolio_df,
                    reason=reason,
                    interactive=False  # Skip confirmation since we already confirmed
                )
                
                # Update portfolio CSV
                portfolio_csv_path = Path(__file__).parent / "Start Your Own" / "chatgpt_portfolio_update.csv"
                print("Updating portfolio CSV...")
                
                # Create the results for CSV update
                today = datetime.now().strftime("%Y-%m-%d")
                results = []
                
                # Add remaining positions from updated_portfolio
                total_value = 0.0
                for _, stock in updated_portfolio.iterrows():
                    stock_value = stock["shares"] * stock["buy_price"]
                    total_value += stock_value
                    results.append({
                        "Date": today,
                        "Ticker": stock["ticker"],
                        "Shares": stock["shares"],
                        "Cost Basis": stock["buy_price"],
                        "Stop Loss": stock["stop_loss"],
                        "Current Price": stock["buy_price"],
                        "Total Value": stock_value,
                        "PnL": 0.0,
                        "Action": "",
                        "Cash Balance": "",
                        "Total Equity": ""
                    })
                
                # Add TOTAL row
                results.append({
                    "Date": today,
                    "Ticker": "TOTAL",
                    "Shares": "",
                    "Cost Basis": "",
                    "Stop Loss": "",
                    "Current Price": "",
                    "Total Value": round(total_value, 2),
                    "PnL": 0.0,
                    "Action": "",
                    "Cash Balance": round(new_cash, 2),
                    "Total Equity": round(total_value + new_cash, 2),
                })
                
                # Save to CSV
                df = pd.DataFrame(results)
                if portfolio_csv_path.exists():
                    existing = pd.read_csv(portfolio_csv_path)
                    existing = existing[existing["Date"] != today]
                    df = pd.concat([existing, df], ignore_index=True)
                
                df.to_csv(portfolio_csv_path, index=False)
                
                # Log decision
                trade_record = {
                    "date": datetime.now().isoformat(),
                    "type": "SELL",
                    "ticker": ticker,
                    "shares": shares,
                    "price": sell_price,
                    "cost_basis": cost_basis,
                    "total_proceeds": total_proceeds,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "reason": reason,
                    "cash_before": cash,
                    "cash_after": new_cash
                }
                
                # Save to trade history
                trade_history = self.load_trade_history()
                trade_history.append(trade_record)
                self.save_trade_history(trade_history)
                
                print(f"✅ Sell trade logged successfully!")
                print(f"   New cash balance: ${new_cash:.2f}")
                print(f"   Realized P&L: ${pnl:+.2f}")
                
                return True
                
            except Exception as e:
                print(f"❌ Error logging trade: {e}")
                return False
                
        except KeyboardInterrupt:
            print("\n❌ Trade logging cancelled")
            return False
    
    def show_recent_decisions(self, days: int = 7):
        """Show recent trading decisions.
        
        Parameters
        ----------
        days : int, optional
            Number of days to look back (default: 7)
        """
        trade_history = self.load_trade_history()
        
        if not trade_history:
            print("📋 No trading decisions recorded yet")
            return
        
        # Filter recent trades
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_trades = []
        
        for trade in trade_history:
            try:
                trade_date = datetime.fromisoformat(trade['date'].replace('Z', '+00:00').replace('+00:00', ''))
                if trade_date >= cutoff_date:
                    recent_trades.append(trade)
            except:
                # Include trades with date parsing issues
                recent_trades.append(trade)
        
        if not recent_trades:
            print(f"📋 No trading decisions in the last {days} days")
            return
        
        print(f"\n📋 RECENT TRADING DECISIONS (Last {days} days)")
        print("=" * 50)
        
        for trade in recent_trades[-10:]:  # Show last 10
            date_str = trade['date'][:10] if 'date' in trade else 'Unknown'
            trade_type = trade.get('type', 'UNKNOWN')
            ticker = trade.get('ticker', 'UNKNOWN')
            shares = trade.get('shares', 0)
            price = trade.get('price', 0)
            reason = trade.get('reason', 'No reason provided')
            
            if trade_type == 'BUY':
                print(f"📈 {date_str}: BUY {shares} {ticker} @ ${price:.2f}")
            elif trade_type == 'SELL':
                pnl = trade.get('pnl', 0)
                print(f"📉 {date_str}: SELL {shares} {ticker} @ ${price:.2f} (P&L: ${pnl:+.2f})")
            
            print(f"   Reason: {reason}")
            print()
    
    def interactive_session(self):
        """Run interactive trading decision logging session."""
        print("💼 DAILY TRADE LOGGER")
        print("=" * 40)
        
        # Show current portfolio
        portfolio_summary = self.get_current_portfolio_summary()
        
        print(f"📊 Current Portfolio:")
        print(f"   Positions: {portfolio_summary['positions']}")
        print(f"   Total Invested: ${portfolio_summary['total_invested']:,.2f}")
        print(f"   Cash Available: ${portfolio_summary['cash_available']:,.2f}")
        print(f"   Total Equity: ${portfolio_summary['total_equity']:,.2f}")
        
        if portfolio_summary['holdings']:
            print(f"\n   Holdings:")
            for holding in portfolio_summary['holdings']:
                print(f"   • {holding['ticker']}: {holding['shares']} shares @ ${holding['cost_basis']:.2f}")
        
        while True:
            print(f"\n" + "─" * 40)
            print("What would you like to do?")
            print("1. Log a BUY trade")
            print("2. Log a SELL trade") 
            print("3. View recent decisions")
            print("4. Show portfolio summary")
            print("5. Exit")
            
            try:
                choice = input("\nEnter choice (1-5): ").strip()
                
                if choice == '1':
                    self.log_buy_trade()
                elif choice == '2':
                    self.log_sell_trade()
                elif choice == '3':
                    days = input("Days to look back (default 7): ").strip()
                    try:
                        days = int(days) if days else 7
                    except ValueError:
                        days = 7
                    self.show_recent_decisions(days)
                elif choice == '4':
                    portfolio_summary = self.get_current_portfolio_summary()
                    print(f"\n📊 PORTFOLIO SUMMARY")
                    print(f"   Positions: {portfolio_summary['positions']}")
                    print(f"   Cash: ${portfolio_summary['cash_available']:,.2f}")
                    print(f"   Total Equity: ${portfolio_summary['total_equity']:,.2f}")
                elif choice == '5':
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-5.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue


def main():
    """Command-line interface for daily trade logger."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Daily Trade Logger")
    parser.add_argument('--buy', action='store_true', help='Log a buy trade')
    parser.add_argument('--sell', action='store_true', help='Log a sell trade')
    parser.add_argument('--recent', type=int, metavar='DAYS', help='Show recent decisions')
    
    args = parser.parse_args()
    
    try:
        logger = DailyTradeLogger()
        
        if args.buy:
            logger.log_buy_trade()
        elif args.sell:
            logger.log_sell_trade()
        elif args.recent is not None:
            logger.show_recent_decisions(args.recent)
        else:
            # Interactive session
            logger.interactive_session()
            
    except Exception as e:
        print(f"Logger failed: {e}")


if __name__ == "__main__":
    main()