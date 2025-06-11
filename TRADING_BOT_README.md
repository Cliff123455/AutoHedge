# E-Trade Paper Trading Bot Setup Guide

## 🎯 Overview

This trading bot combines AI-powered analysis from AutoHedge with automated execution on your E-Trade paper trading account. Perfect for testing strategies risk-free before moving to real money.

## 🛠️ Prerequisites

### 1. E-Trade Developer Account
- Go to [E-Trade Developer Portal](https://developer.etrade.com/)
- Sign in with your E-Trade credentials
- Create a new application to get API credentials
- Note: You'll need both sandbox (paper) and potentially production credentials

### 2. Required API Keys
- **E-Trade API credentials** (Consumer Key, Consumer Secret, OAuth tokens)
- **OpenAI API key** (for AI analysis)
- **Alpha Vantage API key** (optional, for better price data)

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Setup Script
```bash
python setup_etrade_bot.py
```

This will guide you through:
- Setting up your `.env` file with API credentials
- Testing your E-Trade connection
- Ensuring security best practices

### Step 3: Start Trading Bot
```bash
python trading_bot.py
```

Choose from three pre-built strategies:
- **Momentum**: High-risk AI momentum trading
- **Conservative**: Low-risk stable growth 
- **Tech Focus**: Technology sector focused

## 📊 Trading Strategies

### 1. AI Momentum Strategy
- **Stocks**: NVDA, TSLA, AAPL, GOOGL, MSFT
- **Max Position**: $5,000 per stock
- **Risk Level**: 7/10
- **Stop Loss**: 5%
- **Take Profit**: 15%
- **Frequency**: Daily rebalancing

### 2. Conservative Growth Strategy
- **Stocks**: AAPL, MSFT, JNJ, PG, KO
- **Max Position**: $3,000 per stock
- **Risk Level**: 3/10
- **Stop Loss**: 3%
- **Take Profit**: 10%
- **Frequency**: Weekly rebalancing

### 3. Tech Focus Strategy
- **Stocks**: NVDA, AMD, INTC, GOOGL, META
- **Max Position**: $4,000 per stock
- **Risk Level**: 8/10
- **Stop Loss**: 7%
- **Take Profit**: 20%
- **Frequency**: Daily rebalancing

## 🔧 Configuration

### Environment Variables (.env file)
```bash
# E-Trade API Credentials
ETRADE_CONSUMER_KEY=your_consumer_key
ETRADE_CONSUMER_SECRET=your_consumer_secret
ETRADE_OAUTH_TOKEN=your_oauth_token
ETRADE_OAUTH_TOKEN_SECRET=your_oauth_token_secret
ETRADE_ACCOUNT_ID=your_paper_account_id

# AI Analysis
OPENAI_API_KEY=your_openai_key

# Optional: Better price data
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

### Custom Strategy Creation
You can create custom strategies by modifying the `STRATEGIES` dictionary in `trading_bot.py`:

```python
'my_strategy': TradingStrategy(
    name="My Custom Strategy",
    description="Your strategy description",
    stocks=["STOCK1", "STOCK2"],
    max_position_size=5000,
    stop_loss_pct=0.05,
    take_profit_pct=0.15,
    rebalance_frequency='daily',
    risk_level=6
)
```

## 📈 How It Works

### 1. AI Analysis Phase
- The bot uses AutoHedge's AI agents to analyze each stock
- Considers technical indicators, sentiment, and market trends
- Generates BUY/SELL/HOLD recommendations with reasoning

### 2. Decision Making
- Parses AI analysis for actionable recommendations
- Applies risk management rules
- Calculates appropriate position sizes

### 3. Order Execution
- Places orders through E-Trade API
- Logs all trades with detailed information
- Tracks performance metrics

### 4. Monitoring & Reporting
- Continuous monitoring of positions
- Real-time performance tracking
- Detailed logging for analysis

## 📊 Monitoring Your Bot

### Real-time Status
The bot provides detailed status reports including:
- Current positions
- Recent trades
- Performance metrics
- Account balance
- AI analysis results

### Log Files
All activity is logged to `logs/trading_bot_{timestamp}.log`:
- Trade executions
- AI analysis results
- Errors and warnings
- Performance metrics

### Performance Tracking
The bot tracks:
- Total number of trades
- Success rate
- Profit/Loss
- Risk-adjusted returns

## 🛡️ Risk Management

### Built-in Safety Features
- **Paper Trading Default**: Starts in sandbox mode
- **Position Limits**: Maximum position sizes per strategy
- **Stop Losses**: Automatic loss limitation
- **Take Profits**: Automatic profit taking
- **Diversification**: Multiple stock allocation

### Recommended Practices
1. **Start Small**: Use small position sizes initially
2. **Monitor Daily**: Check bot performance regularly
3. **Weekly Reviews**: Analyze strategy effectiveness
4. **Gradual Scaling**: Increase position sizes slowly
5. **Multiple Strategies**: Test different approaches

## 🔄 Running Modes

### Single Execution
```bash
python trading_bot.py
# Choose strategy and run once
```

### Continuous Trading
```bash
python trading_bot.py
# Choose strategy and select "continuous" mode
# Bot will run automatically on schedule
```

### Custom Scheduling
Modify the schedule in `trading_bot.py`:
```python
# Run every hour during market hours
schedule.every().hour.do(bot.execute_trading_cycle)

# Run daily at market open
schedule.every().day.at("09:30").do(bot.execute_trading_cycle)

# Run weekly
schedule.every().monday.at("09:30").do(bot.execute_trading_cycle)
```

## 🚨 Moving to Real Trading

### When You're Ready
After successful paper trading (recommend 1-2 weeks minimum):

1. **Update E-Trade Settings**:
   ```python
   # In trading_bot.py, change:
   paper_trading = False  # Use real account
   ```

2. **Get Production Credentials**:
   - Replace sandbox credentials with production ones
   - Update your `.env` file

3. **Start Conservative**:
   - Use smaller position sizes initially
   - Monitor more closely
   - Gradually increase allocation

### Production Checklist
- ✅ Paper trading profitable for at least 1 week
- ✅ Understand all trades and reasoning
- ✅ Comfortable with strategy risk level
- ✅ Production E-Trade credentials configured
- ✅ Real account funded appropriately
- ✅ Monitoring plan in place

## 🆘 Troubleshooting

### Common Issues

#### Connection Errors
```
Error: Missing E*TRADE credentials
```
**Solution**: Run `python setup_etrade_bot.py` to configure credentials

#### Price Fetching Issues
```
Warning: Could not fetch price for STOCK, using fallback
```
**Solution**: Add Alpha Vantage API key to `.env` file

#### API Rate Limits
```
Error: API request failed
```
**Solution**: The bot includes retry logic and fallbacks

### Getting Help
1. Check log files in `logs/` directory
2. Verify all credentials in `.env` file
3. Test E-Trade connection manually
4. Review E-Trade API documentation

## 📞 Support

### E-Trade API Support
- [E-Trade Developer Documentation](https://developer.etrade.com/home)
- [API Reference](https://developer.etrade.com/docs)

### AutoHedge Documentation
- Check the main README.md in the project root
- Review the AutoHedge documentation

## ⚖️ Legal Disclaimer

This software is for educational and research purposes. Always:
- Test thoroughly in paper trading first
- Understand the risks of automated trading
- Monitor your bot's performance regularly
- Comply with all applicable regulations
- Never risk more than you can afford to lose

**Remember**: Past performance does not guarantee future results. Always do your own research and consider consulting with financial professionals.

## 🔄 Next Steps

1. **Week 1**: Run paper trading, monitor closely
2. **Week 2**: Analyze performance, adjust strategies
3. **Week 3**: Consider real money with small amounts
4. **Ongoing**: Continuously monitor and optimize

Happy trading! 🚀📈 