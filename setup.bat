@echo off
REM AZR Trading Bot Setup Script for Windows
echo 🚀 Setting up KuCoin AZR Self-Learning + Dust Profit Agent

REM Check if Python 3 is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 3 is required but not installed. Please install Python 3.7+
    pause
    exit /b 1
)

REM Check if PM2 is installed
pm2 --version >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing PM2...
    npm install -g pm2
)

REM Create virtual environment
echo 🐍 Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM Install Python dependencies
echo 📦 Installing Python dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo 📁 Creating directory structure...
if not exist logs mkdir logs
if not exist data mkdir data

REM Check environment file
if not exist .azr_env (
    echo ⚠️  Please configure your .azr_env file with KuCoin API credentials
    echo 📝 Template created. Edit .azr_env with your actual credentials.
) else (
    echo ✅ Environment configuration found
)

echo ✅ Setup complete!
echo.
echo 📋 Next steps:
echo 1. Edit .azr_env with your KuCoin API credentials
echo 2. Test with sandbox mode first (SANDBOX_MODE=true)
echo 3. Start the bot with: pm2 start ecosystem.config.json
echo 4. Monitor with: pm2 monit
echo.
echo ⚠️  IMPORTANT: This is for educational purposes. Always test with small amounts!
pause
