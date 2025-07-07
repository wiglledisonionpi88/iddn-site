#!/usr/bin/env python3
"""
Automated Startup Script for KuCoin AZR Trading Bot
Fully autonomous setup and execution
"""

import os
import sys
import time
import subprocess
from datetime import datetime

class AutomatedBotLauncher:
    def __init__(self):
        self.python_exe = "C:/Python313/python.exe"
        self.bot_script = "azr_trading_bot.py"
        self.project_dir = os.getcwd()
        
    def check_environment(self):
        """Check if environment is ready."""
        print("🔧 Checking environment...")
        
        # Check if API credentials are set
        from dotenv import load_dotenv
        load_dotenv('.azr_env')
        
        api_key = os.getenv('API_KEY')
        if not api_key or api_key == 'your_real_kucoin_api_key_here':
            print("⚠️ WARNING: Using placeholder API credentials")
            print("   The bot will run but won't be able to trade")
            print("   Edit .azr_env with real KuCoin API credentials for live trading")
        else:
            print("✅ API credentials configured")
        
        # Check sandbox mode
        sandbox = os.getenv('SANDBOX_MODE', 'true').lower()
        if sandbox == 'false':
            print("🔴 LIVE TRADING MODE ACTIVE - REAL MONEY AT RISK")
        else:
            print("✅ Sandbox mode enabled - Safe testing")
        
        return True
    
    def start_bot_direct(self):
        """Start the bot directly without PM2."""
        print("🚀 Starting KuCoin AZR Trading Bot...")
        print("=" * 50)
        print(f"📅 Started at: {datetime.now()}")
        print("=" * 50)
        
        try:
            # Start the bot process
            cmd = [self.python_exe, self.bot_script]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            print("✅ Bot process started")
            print("📊 Real-time output:")
            print("-" * 30)
            
            # Stream output in real-time
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"🤖 {line.strip()}")
                
                # Check if process is still running
                if process.poll() is not None:
                    break
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code == 0:
                print("✅ Bot completed successfully")
            else:
                print(f"❌ Bot exited with error code: {return_code}")
            
            return return_code
            
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user (Ctrl+C)")
            if 'process' in locals():
                process.terminate()
            return 0
        except Exception as e:
            print(f"❌ Error starting bot: {e}")
            return 1
    
    def create_startup_batch(self):
        """Create a Windows batch file for easy startup."""
        batch_content = f"""@echo off
echo 🤖 KuCoin AZR Trading Bot - Automated Startup
echo ===============================================
cd /d "{self.project_dir}"
{self.python_exe} automated_startup.py
pause
"""
        
        with open('start_bot.bat', 'w') as f:
            f.write(batch_content)
        
        print("✅ Created start_bot.bat for easy startup")
    
    def show_controls(self):
        """Show control instructions."""
        print("\n🎮 BOT CONTROLS:")
        print("=" * 30)
        print("⏹️  Stop Bot: Press Ctrl+C")
        print("📊 Monitor: Check terminal output")
        print("📋 Logs: Check logs/ directory")
        print("📈 Data: Check data/ directory")
        print("🚨 Emergency: Close terminal window")
        print("=" * 30)
    
    def run(self):
        """Main automated startup sequence."""
        print("🚀 AUTOMATED KUCOIN AZR TRADING BOT STARTUP")
        print("=" * 60)
        
        # Environment check
        if not self.check_environment():
            print("❌ Environment check failed")
            return 1
        
        # Create batch file
        self.create_startup_batch()
        
        # Show controls
        self.show_controls()
        
        # Final warning for live trading
        from dotenv import load_dotenv
        load_dotenv('.azr_env')
        sandbox = os.getenv('SANDBOX_MODE', 'true').lower()
        
        if sandbox == 'false':
            print("\n🚨 LIVE TRADING WARNING 🚨")
            print("This bot will trade with REAL MONEY")
            print("Continue? (y/N): ", end='')
            
            try:
                response = input().lower().strip()
                if response != 'y':
                    print("❌ Startup cancelled")
                    return 0
            except KeyboardInterrupt:
                print("\n❌ Startup cancelled")
                return 0
        
        # Start the bot
        print("\n🏁 STARTING BOT...")
        return self.start_bot_direct()

def main():
    """Main entry point."""
    launcher = AutomatedBotLauncher()
    exit_code = launcher.run()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
