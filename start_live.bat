@echo off
REM Live Trading Startup Script for Windows
echo 🚨 KUCOIN AZR BOT - LIVE TRADING MODE
echo ======================================
echo ⚠️  WARNING: REAL MONEY AT RISK ⚠️
echo ======================================

REM Pre-flight safety check
echo 🔍 Running pre-flight safety check...
python live_safety.py check

if errorlevel 1 (
    echo ❌ Safety check failed. Aborting.
    pause
    exit /b 1
)

REM Final confirmation
echo.
echo 🚨 FINAL CONFIRMATION 🚨
echo You are about to start live trading with real money.
echo The bot will:
echo - Trade with your actual KuCoin account
echo - Use real USDT for trades
echo - Consolidate dust automatically
echo - Operate autonomously
echo.
set /p confirmation="Type 'START LIVE TRADING' to continue: "

if not "%confirmation%"=="START LIVE TRADING" (
    echo ❌ Live trading cancelled
    pause
    exit /b 1
)

REM Create safety backup
echo 💾 Creating safety backup...
if not exist backups mkdir backups
copy .azr_env "backups\azr_env_backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"

REM Start the bot
echo 🚀 Starting live trading bot...
pm2 start ecosystem.config.json

if errorlevel 0 (
    echo ✅ Bot started successfully
    echo.
    echo 📊 MONITORING COMMANDS:
    echo - Monitor: python live_safety.py monitor
    echo - Status: pm2 status
    echo - Logs: pm2 logs azr-trading-bot
    echo - EMERGENCY STOP: python live_safety.py stop
    echo.
    echo 🔴 IMPORTANT: Monitor the bot closely!
    echo.
    echo Press any key to start safety monitor...
    pause >nul
    
    REM Start safety monitor
    echo 🛡️ Starting safety monitor...
    python live_safety.py monitor
    
) else (
    echo ❌ Failed to start bot
    pause
    exit /b 1
)
