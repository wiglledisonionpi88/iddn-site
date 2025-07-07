#!/usr/bin/env python3
"""
Enhanced Live Trading Launcher
Debug, update, and run with real cash
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from dotenv import load_dotenv

class LiveTradingLauncher:
    def __init__(self):
        self.python_exe = "C:/Python313/python.exe"
        self.project_dir = os.getcwd()
        
    def debug_system(self):
        """Debug and check system status."""
        print("🔍 DEBUGGING SYSTEM...")
        print("=" * 50)
        
        issues = []
        
        # Check environment file
        if not os.path.exists('.azr_env'):
            issues.append("❌ .azr_env file missing")
        else:
            from dotenv import load_dotenv
            load_dotenv('.azr_env')
            api_key = os.getenv('API_KEY', '')
            if api_key == 'your_real_kucoin_api_key_here':
                issues.append("⚠️ Placeholder API credentials detected")
            else:
                print("✅ Real API credentials configured")
        
        # Check dependencies
        try:
            import ccxt
            import numpy
            from dotenv import load_dotenv
            print("✅ All dependencies available")
        except ImportError as e:
            issues.append(f"❌ Missing dependency: {e}")
        
        # Check directories
        for dir_name in ['logs', 'data']:
            if os.path.exists(dir_name):
                print(f"✅ Directory exists: {dir_name}")
            else:
                print(f"📁 Creating directory: {dir_name}")
                os.makedirs(dir_name)
        
        # Check bot file
        if os.path.exists('azr_trading_bot.py'):
            print("✅ Bot script found")
        else:
            issues.append("❌ azr_trading_bot.py not found")
        
        return issues
    
    def update_for_live_trading(self):
        """Update configuration for live trading."""
        print("🔄 UPDATING FOR LIVE TRADING...")
        
        from dotenv import load_dotenv
        load_dotenv('.azr_env')
        
        # Check current config
        sandbox = os.getenv('SANDBOX_MODE', 'true').lower()
        api_key = os.getenv('API_KEY', '')
        
        print(f"Current sandbox mode: {sandbox}")
        print(f"API key configured: {'Yes' if api_key != 'your_real_kucoin_api_key_here' else 'No (placeholder)'}")
        
        if api_key == 'your_real_kucoin_api_key_here':
            print("\n🔑 REAL API CREDENTIALS REQUIRED")
            print("Please edit .azr_env with your actual KuCoin API credentials:")
            print("1. Go to https://www.kucoin.com/account/api")
            print("2. Create API key with trading permissions")
            print("3. Copy credentials to .azr_env file")
            return False
        
        return True
    
    def kill_existing_bots(self):
        """Kill any existing bot processes."""
        try:
            # Kill Python processes running the bot
            subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], 
                          capture_output=True, check=False)
            print("🔄 Cleared existing bot processes")
            time.sleep(2)
        except:
            pass
    
    def start_live_bot(self):
        """Start the bot for live trading."""
        print("🚀 STARTING LIVE TRADING BOT...")
        print("=" * 50)
        
        # Kill existing processes
        self.kill_existing_bots()
        
        # Start bot in background
        cmd = [self.python_exe, 'azr_trading_bot.py']
        
        try:
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
            
            print("✅ Bot process started")
            print(f"📊 Process ID: {process.pid}")
            
            # Give it a moment to start
            time.sleep(3)
            
            # Check if it's still running
            if process.poll() is None:
                print("✅ Bot is running successfully")
                return process
            else:
                print("❌ Bot exited immediately")
                return None
                
        except Exception as e:
            print(f"❌ Failed to start bot: {e}")
            return None
    
    def monitor_bot(self, process):
        """Monitor the bot and show status."""
        print("\n📊 MONITORING BOT...")
        print("=" * 50)
        print("🎮 CONTROLS:")
        print("- Press 'q' + Enter to stop bot")
        print("- Press 's' + Enter to show status")
        print("- Press 'l' + Enter to show latest logs")
        print("=" * 50)
        
        try:
            while True:
                # Check if process is still running
                if process.poll() is not None:
                    print("❌ Bot process has stopped")
                    break
                
                # Check for user input
                import select
                import sys
                
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    user_input = sys.stdin.readline().strip().lower()
                    
                    if user_input == 'q':
                        print("🛑 Stopping bot...")
                        process.terminate()
                        break
                    elif user_input == 's':
                        self.show_status()
                    elif user_input == 'l':
                        self.show_latest_logs()
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Ctrl+C pressed - stopping bot")
            process.terminate()
    
    def show_status(self):
        """Show current bot status."""
        print("\n" + "="*30)
        print(f"📊 STATUS at {datetime.now().strftime('%H:%M:%S')}")
        
        # Check if process is running
        try:
            result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                                   capture_output=True, text=True)
            if 'python.exe' in result.stdout:
                print("🟢 Bot process: RUNNING")
            else:
                print("🔴 Bot process: NOT FOUND")
        except:
            print("❓ Bot process: UNKNOWN")
        
        # Check log file
        today = datetime.now().strftime('%Y%m%d')
        log_file = f"logs/azr_bot_{today}.log"
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    print(f"📋 Log entries: {len(lines)}")
                    if lines:
                        last_line = lines[-1].strip()
                        print(f"📝 Last log: {last_line[-50:]}")  # Last 50 chars
            except:
                print("❌ Error reading logs")
        else:
            print("📋 No log file found")
        
        print("="*30 + "\n")
    
    def show_latest_logs(self):
        """Show latest log entries."""
        today = datetime.now().strftime('%Y%m%d')
        log_file = f"logs/azr_bot_{today}.log"
        
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    print("\n📋 LATEST LOGS:")
                    print("-" * 40)
                    for line in lines[-5:]:  # Last 5 lines
                        print(line.strip())
                    print("-" * 40 + "\n")
            except Exception as e:
                print(f"❌ Error reading logs: {e}")
        else:
            print("📋 No log file found")
    
    def run(self):
        """Main run sequence."""
        print("🚀 LIVE TRADING LAUNCHER")
        print("=" * 60)
        print("⚠️  WARNING: REAL MONEY AT RISK")
        print("=" * 60)
        
        # Debug system
        issues = self.debug_system()
        if issues:
            print("\n❌ ISSUES FOUND:")
            for issue in issues:
                print(f"   {issue}")
            print("\nPlease fix these issues and try again.")
            return 1
        
        # Update for live trading
        if not self.update_for_live_trading():
            print("\n❌ Configuration incomplete")
            return 1
        
        # Final confirmation
        print("\n🚨 FINAL CONFIRMATION")
        print("You are about to start LIVE TRADING with REAL MONEY")
        response = input("Type 'START LIVE TRADING' to continue: ").strip()
        
        if response != 'START LIVE TRADING':
            print("❌ Live trading cancelled")
            return 0
        
        # Start bot
        process = self.start_live_bot()
        if not process:
            print("❌ Failed to start bot")
            return 1
        
        # Monitor bot
        self.monitor_bot(process)
        
        print("✅ Live trading session ended")
        return 0

def main():
    """Main entry point."""
    launcher = LiveTradingLauncher()
    exit_code = launcher.run()
    input("Press Enter to exit...")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
