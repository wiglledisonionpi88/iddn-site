#!/bin/bash

# AZR Trading Bot Setup Script
echo "🚀 Setting up KuCoin AZR Self-Learning + Dust Profit Agent"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.7+"
    exit 1
fi

# Check if PM2 is installed
if ! command -v pm2 &> /dev/null; then
    echo "📦 Installing PM2..."
    npm install -g pm2
fi

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p logs
mkdir -p data

# Set up environment file
if [ ! -f .azr_env ]; then
    echo "⚠️  Please configure your .azr_env file with KuCoin API credentials"
    echo "📝 Template created. Edit .azr_env with your actual credentials."
else
    echo "✅ Environment configuration found"
fi

# Make the bot script executable
chmod +x azr_trading_bot.py

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .azr_env with your KuCoin API credentials"
echo "2. Test with sandbox mode first (SANDBOX_MODE=true)"
echo "3. Start the bot with: pm2 start ecosystem.config.json"
echo "4. Monitor with: pm2 monit"
echo ""
echo "⚠️  IMPORTANT: This is for educational purposes. Always test with small amounts!"
