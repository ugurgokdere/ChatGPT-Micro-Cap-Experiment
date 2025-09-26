# Project Structure - AI Trading System

## Active Files (Core System)

### Main Scripts (What You Run)
- **`interactive_research_analyzer.py`** - Weekly research analysis tool
  - Run weekly with PDF research
  - Analyzes stocks and provides BUY/SELL recommendations
  - Now supports ALL market cap sizes (not just micro-caps)

- **`daily_trade_logger.py`** - Daily trade logging system
  - Run daily to log your actual trades
  - Keeps portfolio synchronized with reality

### Core Dependencies (Required)
- **`ai_analyst.py`** - OpenAI integration for analysis
- **`market_research.py`** - Finnhub API for market data
- **`portfolio_analyzer.py`** - Portfolio state management
- **`trading_script.py`** - Core trading functions & CSV management

### Data Files
- **`Start Your Own/chatgpt_portfolio_update.csv`** - Current positions
- **`Start Your Own/chatgpt_trade_log.csv`** - Trade history
- **`Start Your Own/Generate_Graph.py`** - Performance visualization
- **`weekly_analysis/`** - Stored analysis results
- **`daily_trade_decisions.json`** - Trade decision log

### Configuration
- **`env.local`** - API keys (FINNHUB_API_KEY, OPENAI_API_KEY)
- **`requirements.txt`** - Python dependencies
- **`CLAUDE.md`** - Instructions for Claude AI

## Archived Files
All legacy/unused files have been moved to `archive/legacy_files/`:
- api_server.py
- n8n_wrapper.py
- update_portfolio_prices.py
- N8N_SETUP.md
- n8n_trading_workflow.json
- watchlist.txt
- Old "Scripts and CSV Files" directory
- Old "Weekly Deep Research" directories

## Quick Commands

```bash
# Weekly analysis (input stocks from research)
python interactive_research_analyzer.py

# Daily trade logging
python daily_trade_logger.py

# View performance
python "Start Your Own/Generate_Graph.py" --baseline-equity 100
```