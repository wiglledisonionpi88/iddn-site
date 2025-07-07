#!/bin/bash

# Live Trading Startup Script
echo "🚨 KUCOIN AZR BOT - LIVE TRADING MODE"
echo "======================================"
echo "⚠️  WARNING: REAL MONEY AT RISK ⚠️"
echo "======================================"

# Pre-flight safety check
echo "🔍 Running pre-flight safety check..."
python3 live_safety.py check

if [ $? -ne 0 ]; then
    echo "❌ Safety check failed. Aborting."
    exit 1
fi

# Final confirmation
echo ""
echo "🚨 FINAL CONFIRMATION 🚨"
echo "You are about to start live trading with real money."
echo "The bot will:"
echo "- Trade with your actual KuCoin account"
echo "- Use real USDT for trades"
echo "- Consolidate dust automatically"
echo "- Operate autonomously"
echo ""
read -p "Type 'START LIVE TRADING' to continue: " confirmation

if [ "$confirmation" != "START LIVE TRADING" ]; then
    echo "❌ Live trading cancelled"
    exit 1
fi

# Create safety backup
echo "💾 Creating safety backup..."
mkdir -p backups
cp .azr_env "backups/azr_env_backup_$(date +%Y%m%d_%H%M%S)"

# Start the bot
echo "🚀 Starting live trading bot..."
pm2 start ecosystem.config.json

if [ $? -eq 0 ]; then
    echo "✅ Bot started successfully"
    echo ""
    echo "📊 MONITORING COMMANDS:"
    echo "- Monitor: python3 live_safety.py monitor"
    echo "- Status: pm2 status"
    echo "- Logs: pm2 logs azr-trading-bot"
    echo "- EMERGENCY STOP: python3 live_safety.py stop"
    echo ""
    echo "🔴 IMPORTANT: Monitor the bot closely!"
    
    # Start safety monitor in background
    echo "🛡️ Starting safety monitor..."
    python3 live_safety.py monitor &
    
else
    echo "❌ Failed to start bot"
    exit 1
fi
