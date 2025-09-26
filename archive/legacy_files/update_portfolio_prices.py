#!/usr/bin/env python3
"""Update portfolio CSV with current market prices."""

from market_research import MarketResearcher
import pandas as pd
from pathlib import Path

def update_portfolio_prices():
    """Fetch current prices and update portfolio CSV."""
    # Get current prices for portfolio positions
    researcher = MarketResearcher()
    
    # Read current portfolio
    portfolio_path = Path('Start Your Own/chatgpt_portfolio_update.csv')
    if not portfolio_path.exists():
        print("Portfolio CSV not found!")
        return
        
    df = pd.read_csv(portfolio_path)
    
    # Get unique tickers (excluding TOTAL)
    tickers = df[df['Ticker'] != 'TOTAL']['Ticker'].unique()
    
    print("Fetching current prices...")
    current_prices = {}
    
    for ticker in tickers:
        try:
            fundamentals = researcher.get_stock_fundamentals(ticker)
            current_price = fundamentals.get('current_price', 0)
            current_prices[ticker] = current_price
            print(f'{ticker}: Current price = ${current_price:.4f}')
        except Exception as e:
            print(f'{ticker}: Error getting price - {e}')
    
    print("\nUpdating portfolio...")
    
    # Get today's date
    from datetime import datetime
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Update current prices and recalculate Total Value and PnL ONLY for today's rows
    for idx, row in df.iterrows():
        ticker = row['Ticker']
        date = row['Date']
        
        # Only update today's entries
        if date == today and ticker in current_prices and current_prices[ticker] > 0:
            shares = row['Shares']
            cost_basis = row['Cost Basis']
            current_price = current_prices[ticker]
            
            # Update current price
            df.at[idx, 'Current Price'] = current_price
            
            # Recalculate total value
            new_total_value = shares * current_price
            df.at[idx, 'Total Value'] = round(new_total_value, 2)
            
            # Calculate P&L
            original_cost = shares * cost_basis
            pnl = new_total_value - original_cost
            df.at[idx, 'PnL'] = round(pnl, 2)
            
            pnl_pct = (pnl / original_cost) * 100 if original_cost > 0 else 0
            print(f'  {ticker}: {shares:.2f} shares @ ${cost_basis:.4f} → ${current_price:.4f} = ${new_total_value:.2f} (P&L: ${pnl:+.2f}, {pnl_pct:+.1f}%)')
    
    # Update TOTAL rows for each date
    for date in df['Date'].unique():
        total_rows = df[(df['Date'] == date) & (df['Ticker'] == 'TOTAL')]
        if len(total_rows) > 0:
            total_idx = total_rows.index[0]
            
            # Calculate total portfolio value for this date
            date_positions = df[(df['Date'] == date) & (df['Ticker'] != 'TOTAL')]
            
            if len(date_positions) > 0:
                total_value = date_positions['Total Value'].sum()
                total_pnl = date_positions['PnL'].sum()
            else:
                total_value = 0
                total_pnl = 0
            
            cash = df.at[total_idx, 'Cash Balance']
            
            df.at[total_idx, 'Total Value'] = round(total_value, 2)
            df.at[total_idx, 'PnL'] = round(total_pnl, 2)
            df.at[total_idx, 'Total Equity'] = round(total_value + cash, 2)
    
    # Save updated CSV
    df.to_csv(portfolio_path, index=False)
    
    # Show final summary for latest date
    latest_date = df['Date'].max()
    latest_total = df[(df['Date'] == latest_date) & (df['Ticker'] == 'TOTAL')]
    
    if len(latest_total) > 0:
        row = latest_total.iloc[0]
        print(f'\nPortfolio Summary ({latest_date}):')
        print(f'  Total Value: ${row["Total Value"]:.2f}')
        print(f'  Total P&L: ${row["PnL"]:+.2f}')
        print(f'  Cash: ${row["Cash Balance"]:.2f}')
        print(f'  Total Equity: ${row["Total Equity"]:.2f}')
    
    print("\n✅ Portfolio CSV updated with current prices!")

if __name__ == "__main__":
    update_portfolio_prices()