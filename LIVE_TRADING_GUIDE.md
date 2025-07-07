# 🚨 LIVE TRADING SETUP GUIDE 🚨

## ⚠️ CRITICAL WARNING ⚠️
**YOU ARE ABOUT TO CONFIGURE LIVE TRADING WITH REAL MONEY**
- Cryptocurrency trading involves substantial risk of loss
- This bot can lose money quickly in volatile markets
- Always start with small amounts you can afford to lose
- Monitor the bot closely during operation

## 📋 LIVE TRADING CHECKLIST

### 1. 🔑 Configure Real API Credentials
Edit `.azr_env` and replace the placeholder values:
```env
API_KEY=your_real_kucoin_api_key_here
API_SECRET=your_real_kucoin_api_secret_here
API_PASSPHRASE=your_real_kucoin_api_passphrase_here
```

**Get your KuCoin API credentials:**
1. Go to https://www.kucoin.com/account/api
2. Create new API key with trading permissions
3. **IMPORTANT**: Restrict IP access to your server's IP
4. Copy the credentials to `.azr_env`

### 2. ⚙️ Safety Configuration
The bot is now configured with conservative settings:
- `MIN_TRADE_AMOUNT=10` - Minimum $10 per trade
- `TRADING_PAIRS_COUNT=5` - Only 5 trading pairs (reduced for safety)
- `TRADING_CYCLE_MINUTES=10` - 10-minute cycles (slower for safety)
- `MOMENTUM_THRESHOLD=0.15` - Higher threshold for more conservative trading
- `SANDBOX_MODE=false` - ⚠️ LIVE TRADING ENABLED

### 3. 🚀 Starting Live Trading

#### Windows:
```cmd
# Run safety check first
python test_bot.py

# Start live trading (with confirmations)
start_live.bat
```

#### Linux/macOS:
```bash
# Run safety check first
python3 test_bot.py

# Start live trading (with confirmations)
./start_live.sh
```

### 4. 🛡️ Safety Controls

#### Emergency Stop:
```cmd
python live_safety.py stop
```

#### Monitor Bot:
```cmd
python live_safety.py monitor
```

#### Check Status:
```cmd
pm2 status
pm2 logs azr-trading-bot
```

### 5. 📊 Monitoring Commands

```cmd
# Real-time monitoring
python monitor.py loop

# PM2 monitoring
pm2 monit

# View logs
pm2 logs azr-trading-bot --lines 50

# Check performance
type data\performance_*.json
```

## 🚨 EMERGENCY PROCEDURES

### Immediate Stop:
1. Run: `python live_safety.py stop`
2. Or: `pm2 stop azr-trading-bot`
3. Or: Create file named `EMERGENCY_STOP` in bot directory

### If Bot Becomes Unresponsive:
1. `pm2 kill` (stops all PM2 processes)
2. Task Manager → End Python processes
3. Check your KuCoin account manually

## 📈 Expected Behavior

The bot will:
1. **Analyze** top 5 trading pairs every 10 minutes
2. **Consolidate** dust amounts (<$1) into USDT automatically
3. **Trade** only when momentum > 0.15% with volume confirmation
4. **Log** all activities to `logs/` directory
5. **Save** performance data to `data/` directory

## ⚠️ RISK FACTORS

- **Market Volatility**: Crypto markets can move rapidly
- **API Failures**: Exchange downtime can affect trading
- **Network Issues**: Connection problems may cause delays
- **Bug Risk**: Software bugs could cause unexpected behavior

## 💡 RECOMMENDED APPROACH

1. **Start Small**: Use only $100-500 initially
2. **Monitor Closely**: Watch the first few hours/days
3. **Check Regularly**: Review logs and performance data
4. **Adjust Settings**: Modify `.azr_env` based on performance
5. **Scale Gradually**: Increase amounts only after successful operation

## 📞 SUPPORT

If you encounter issues:
- Check logs in `logs/` directory
- Review error messages in terminal
- Verify API credentials and permissions
- Ensure sufficient USDT balance in account

## 🎯 SUCCESS INDICATORS

- Bot connects without errors
- Trades execute successfully
- Dust consolidation works
- Performance data is generated
- No critical errors in logs

---

**REMEMBER: Only trade with money you can afford to lose!**
