# KuCoin AZR Self-Learning + Dust Profit Agent

![Trading Bot Status](https://img.shields.io/badge/Status-Active-green)
![Python Version](https://img.shields.io/badge/Python-3.7+-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

A sophisticated cryptocurrency trading bot that performs automated trading on KuCoin with self-learning capabilities and dust consolidation features.

## 🚀 Features

### 🔐 Environment Setup
- ✅ Load environment variables from `.azr_env` file
- ✅ Install required dependencies (ccxt, numpy)
- ✅ Create trading logic directory structure

### 📈 Trading Bot Functionality
- ✅ **Exchange Integration**: KuCoin API connection with proper authentication
- ✅ **Dust Consolidation**: Automatically sell small coin amounts (<$1) to USDT
- ✅ **Active Pair Selection**: Monitor top 10 trading pairs by volume
- ✅ **Automated Trading**: Buy/sell based on price movement analysis

### 🧠 Self-Learning Mechanisms
- ✅ Analyze 1-minute OHLCV data for trend detection
- ✅ Calculate percentage changes and volatility
- ✅ Make trading decisions based on momentum (>0.1% or <-0.1%)
- ✅ Estimate profit potential for each trade

### 🛡️ Risk Management
- ✅ Equal distribution of USDT across selected pairs
- ✅ Minimum trade amounts to prevent micro-transactions
- ✅ Comprehensive error logging and recovery

### 🔄 Process Management
- ✅ Continuous operation with 5-minute cycles
- ✅ JSON logging of all trades and dust operations
- ✅ PM2 process management for reliability
- ✅ Real-time performance monitoring

## 📋 Technical Requirements

- **Python 3.7+** with ccxt and numpy libraries
- **KuCoin API credentials** (API_KEY, API_SECRET, API_PASSPHRASE)
- **PM2** for process management
- **Unix/Linux environment** compatibility (Windows batch scripts included)

## 🔧 Installation

### Quick Setup (Linux/macOS)
```bash
chmod +x setup.sh
./setup.sh
```

### Quick Setup (Windows)
```cmd
setup.bat
```

### Manual Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p logs data

# Configure environment
cp .azr_env.example .azr_env
# Edit .azr_env with your KuCoin API credentials
```

## ⚙️ Configuration

Edit `.azr_env` with your KuCoin API credentials:

```env
# KuCoin API Configuration
API_KEY=your_kucoin_api_key_here
API_SECRET=your_kucoin_api_secret_here
API_PASSPHRASE=your_kucoin_api_passphrase_here

# Trading Configuration
MIN_TRADE_AMOUNT=10
MAX_DUST_VALUE=1.0
TRADING_PAIRS_COUNT=10
TRADING_CYCLE_MINUTES=5
MOMENTUM_THRESHOLD=0.1

# Environment
SANDBOX_MODE=true  # Set to false for live trading
LOG_LEVEL=INFO
```

## 🚀 Usage

### Paper Trading (Recommended First)
```bash
python paper_trading.py
```

### Live Trading with PM2
```bash
# Start the bot
pm2 start ecosystem.config.json

# Monitor the bot
pm2 monit

# View logs
pm2 logs azr-trading-bot

# Stop the bot
pm2 stop azr-trading-bot
```

### Direct Python Execution
```bash
python azr_trading_bot.py
```

## 📊 Monitoring

The bot generates comprehensive logs and performance data:

- **Logs**: `logs/azr_bot_YYYYMMDD.log`
- **Performance Data**: `data/performance_YYYYMMDD_HHMMSS.json`
- **Paper Trading Results**: `data/paper_trading_YYYYMMDD_HHMMSS.json`

### PM2 Monitoring Commands
```bash
pm2 status          # Check bot status
pm2 logs            # View real-time logs
pm2 monit           # Interactive monitoring
pm2 restart all     # Restart bot
```

## 🔍 Trading Algorithm

The bot uses a sophisticated self-learning algorithm:

1. **Market Analysis**: Fetches 1-minute OHLCV data for top trading pairs
2. **Trend Detection**: Calculates price changes, volatility, and volume ratios
3. **Signal Generation**: Generates BUY/SELL/HOLD signals based on momentum
4. **Risk Assessment**: Evaluates confidence levels before executing trades
5. **Position Sizing**: Equal distribution of USDT across selected pairs
6. **Dust Management**: Automatically consolidates small balances

### Signal Logic
- **BUY Signal**: Price change > +0.1% with volume spike (>1.2x average)
- **SELL Signal**: Price change < -0.1% with volume spike (>1.2x average)
- **HOLD Signal**: Insufficient momentum or volume

## 🔒 Security Considerations

- ✅ Environment variable storage for API credentials
- ✅ Rate limiting compliance
- ✅ Error handling for API failures
- ✅ Secure logging without credential exposure
- ✅ Sandbox mode for testing

## 📈 Performance Tracking

The bot tracks:
- Individual trade performance
- Dust consolidation efficiency
- Market analysis accuracy
- Overall portfolio performance
- Error rates and recovery times

## ⚠️ Important Disclaimers

**This bot is for educational purposes only. Cryptocurrency trading involves substantial risk of loss.**

- Always test with small amounts first
- Use sandbox mode before live trading
- Understand the risks of automated trading
- Monitor the bot regularly
- Keep API credentials secure

## 🛠️ Implementation Tasks

- [x] Create environment configuration
- [x] Implement KuCoin API integration
- [x] Build dust consolidation logic
- [x] Develop trading algorithms
- [x] Add comprehensive logging
- [x] Set up PM2 process management
- [x] Implement paper trading mode
- [x] Add error recovery mechanisms

## ✅ Success Criteria

- [x] Bot connects to KuCoin successfully
- [x] Dust consolidation works automatically
- [x] Trading decisions are logged and executed
- [x] Process runs continuously without manual intervention
- [x] Learning log captures performance data

## 📚 File Structure

```
├── azr_trading_bot.py          # Main trading bot
├── paper_trading.py            # Paper trading mode
├── requirements.txt            # Python dependencies
├── .azr_env                    # Environment configuration
├── ecosystem.config.json       # PM2 configuration
├── setup.sh                    # Linux/macOS setup script
├── setup.bat                   # Windows setup script
├── logs/                       # Log files
├── data/                       # Performance data
└── README.md                   # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test thoroughly with paper trading
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For issues and questions:
- 📧 Email: sam.hiotis@gmail.com
- 💻 GitHub: [wiglledisonionpi88](https://github.com/wiglledisonionpi88)

---

**Happy Trading! 🚀📈**
