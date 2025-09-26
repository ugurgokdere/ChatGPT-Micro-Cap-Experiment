# N8N Trading Automation Setup

This document explains how to set up n8n to automate your trading analysis and receive email notifications for buy/sell/hold decisions.

## Overview

The n8n workflow automates:
- Daily execution of trading analysis
- Stock watchlist analysis with AI recommendations
- Gmail notifications for trading decisions
- Error handling and logging

## Prerequisites

1. **n8n installed** (self-hosted or cloud)
2. **Gmail OAuth2 credentials** configured in n8n
3. **Python environment** with all project dependencies
4. **API keys** configured in `env.local`

## Installation Steps

### 1. Install n8n

**Option A: Docker (Recommended)**
```bash
docker run -it --rm --name n8n -p 5678:5678 -v ~/.n8n:/home/node/.n8n n8nio/n8n
```

**Option B: npm**
```bash
npm install n8n -g
n8n start
```

### 2. Configure Gmail OAuth2

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Gmail API
4. Create OAuth2 credentials (Web application)
5. Add authorized redirect URI: `http://localhost:5678/rest/oauth2-credential/callback`
6. In n8n:
   - Go to Settings → Credentials
   - Add "Gmail OAuth2" credential
   - Enter Client ID and Client Secret
   - Complete OAuth flow

### 3. Import Workflow

1. In n8n interface, click "Import from File"
2. Select `n8n_trading_workflow.json`
3. Update the following nodes:

#### Schedule Trigger Node
- Set desired execution time (default: daily at 9:30 AM EST)
- Timezone: America/New_York

#### Execute Command Nodes
- Update path to your project directory
- Ensure Python environment is accessible

#### Gmail Nodes
- Update `toEmail` with your email address
- Update `fromEmail` if needed
- Assign Gmail OAuth2 credential

### 4. Configure Watchlist

Edit `watchlist.txt` with your preferred stock tickers:
```
AAPL
MSFT
GOOGL
TSLA
NVDA
AMD
META
AMZN
```

### 5. Test Setup

**Test the wrapper script:**
```bash
python n8n_wrapper.py analyze --tickers AAPL MSFT --output json
```

**Test email output:**
```bash
python n8n_wrapper.py analyze --watchlist-file watchlist.txt --output email
```

**Test workflow in n8n:**
1. Click "Test workflow" in n8n interface
2. Check email delivery
3. Verify error handling

## Workflow Components

### 1. Schedule Trigger
- **Purpose**: Runs daily at market open
- **Configuration**: 24-hour interval, EST timezone
- **Notes**: Adjust time based on your trading schedule

### 2. Execute Analysis
- **Purpose**: Runs trading analysis script
- **Command**: `python n8n_wrapper.py analyze --watchlist-file watchlist.txt --output json`
- **Output**: JSON with buy/sell/hold recommendations

### 3. Parse Results
- **Purpose**: Converts script output to JSON
- **Function**: Parses stdout from command execution
- **Error Handling**: Catches parsing errors

### 4. Check Success
- **Purpose**: Routes successful vs error results
- **Condition**: `status === "success"`
- **Branches**: Success → Email formatting, Error → Error handling

### 5. Format Email Content
- **Purpose**: Prepares HTML/text email content
- **Features**:
  - Priority setting (high for action items)
  - HTML formatting for recommendations
  - Action summary in subject line

### 6. Send Gmail Notification
- **Purpose**: Delivers trading recommendations via email
- **Features**:
  - HTML email with formatted recommendations
  - High priority for action items
  - Professional formatting

### 7. Action Check
- **Purpose**: Determines if immediate action is needed
- **Condition**: `buyCount > 0 || sellCount > 0`
- **Follow-up**: Logs decisions for audit trail

### 8. Error Handling
- **Purpose**: Handles analysis failures
- **Actions**:
  - Sends error notification email
  - Logs error details
  - High priority alert

## Email Notifications

### Buy/Sell Alert Example
```
Subject: 🚨 ACTION REQUIRED: 2 BUY, 1 SELL signals

- BUY: AAPL @ $150.25 (HIGH confidence)
- BUY: MSFT @ $310.50 (MEDIUM confidence)  
- SELL: TSLA (Not in current research focus)
```

### Hold-Only Example
```
Subject: 📊 Portfolio Update: All positions on HOLD

No action required - all positions on hold based on current analysis.
```

### Error Alert Example
```
Subject: ⚠️ Trading Analysis Error

The trading analysis failed. Please check system logs.
```

## Customization Options

### Schedule Changes
Modify the Schedule Trigger node:
- **Daily at market close**: Set to 4:00 PM EST
- **Multiple times per day**: Add additional Schedule nodes
- **Weekly only**: Change interval to weekly
- **Market days only**: Add day-of-week conditions

### Watchlist Management
- **Static list**: Edit `watchlist.txt`
- **Dynamic from CSV**: Modify command to read from portfolio
- **API integration**: Add API call node to fetch watchlist

### Email Customization
Modify the Format Email node:
- **Different recipients**: Add multiple email addresses
- **Slack notifications**: Replace Gmail with Slack node
- **SMS alerts**: Add SMS node for high-priority items
- **Discord/Teams**: Use webhook nodes

### Analysis Parameters
Update the Execute Command node:
- **Different timeframes**: Add `--timeframe` parameter
- **Risk settings**: Add `--risk-level` parameter
- **Portfolio size limits**: Add `--max-position` parameter

## Security Considerations

### API Keys
- Store in n8n environment variables
- Never commit keys to version control
- Rotate keys regularly

### Email Security
- Use OAuth2 (not passwords)
- Limit Gmail API permissions
- Monitor API usage

### Server Security
- Use HTTPS for n8n access
- Secure server access
- Regular backups of workflows

## Troubleshooting

### Common Issues

**"Command not found" error:**
- Verify Python path in Execute Command node
- Check virtual environment activation
- Ensure all dependencies installed

**Gmail authentication failed:**
- Re-authenticate OAuth2 credential
- Check API quotas in Google Console
- Verify redirect URI configuration

**Analysis returns errors:**
- Check API key configuration in `env.local`
- Verify internet connectivity
- Test wrapper script manually

**No email received:**
- Check spam folder
- Verify email address in Gmail node
- Check n8n execution logs

### Debug Steps

1. **Test wrapper script separately:**
   ```bash
   python n8n_wrapper.py analyze --tickers AAPL --output json
   ```

2. **Check n8n execution logs:**
   - Go to Executions tab in n8n
   - Click on failed execution
   - Review each node's output

3. **Verify credentials:**
   - Test Gmail credential in n8n
   - Verify API keys in environment

4. **Monitor system resources:**
   - Check disk space
   - Monitor memory usage
   - Verify network connectivity

## Monitoring & Maintenance

### Regular Tasks
- Review execution logs weekly
- Update watchlist monthly
- Rotate API keys quarterly
- Backup workflow configurations

### Performance Optimization
- Monitor execution times
- Optimize watchlist size
- Cache market data when possible
- Use error retry mechanisms

### Scaling Options
- Multiple watchlists for different strategies
- Portfolio-specific workflows
- Regional market workflows
- Risk management integration

## Support

For issues with:
- **n8n setup**: Check [n8n documentation](https://docs.n8n.io/)
- **Gmail API**: Review [Google API documentation](https://developers.google.com/gmail/api)
- **Python scripts**: Check project logs and error messages
- **Trading analysis**: Review AI analysis output in logs