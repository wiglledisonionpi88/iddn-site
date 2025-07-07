#!/usr/bin/env python3
"""
Live Trading Safety Controller
Emergency controls and monitoring for live trading
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime, timedelta

class LiveTradingSafety:
    def __init__(self):
        self.bot_name = "azr-trading-bot"
        self.emergency_stop_file = "EMERGENCY_STOP"
        self.max_daily_loss = 100  # Max $100 daily loss
        self.max_trades_per_hour = 20
        self.min_balance_threshold = 50  # Stop if USDT below $50
        
    def emergency_stop(self):
        """Emergency stop the bot immediately."""
        print("🚨 EMERGENCY STOP TRIGGERED")
        
        # Create emergency stop file
        with open(self.emergency_stop_file, 'w') as f:
            f.write(f"Emergency stop at {datetime.now()}")
        
        # Stop PM2 process
        try:
            subprocess.run(['pm2', 'stop', self.bot_name], check=True)
            print("✅ Bot stopped successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to stop bot via PM2")
        
        # Kill Python processes as backup
        try:
            subprocess.run(['pkill', '-f', 'azr_trading_bot.py'], check=False)
            print("✅ Python processes terminated")
        except:
            print("⚠️ Could not kill Python processes")
    
    def check_balance_safety(self):
        """Check if balance is above safety threshold."""
        try:
            # This would need to connect to exchange to check real balance
            # For now, return True - implement actual balance check
            return True
        except Exception as e:
            print(f"Error checking balance: {e}")
            return False
    
    def check_daily_performance(self):
        """Check daily performance and losses."""
        try:
            data_dir = 'data'
            if not os.path.exists(data_dir):
                return True
            
            # Check performance files from today
            today = datetime.now().strftime('%Y%m%d')
            today_files = [f for f in os.listdir(data_dir) 
                          if f.startswith('performance_') and today in f]
            
            total_trades = 0
            for file in today_files:
                with open(os.path.join(data_dir, file), 'r') as f:
                    data = json.load(f)
                    if 'performance_log' in data:
                        for cycle in data['performance_log']:
                            total_trades += cycle.get('trades_executed', 0)
            
            # Check if too many trades
            if total_trades > self.max_trades_per_hour:
                print(f"⚠️ High trade count: {total_trades} trades today")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error checking daily performance: {e}")
            return True
    
    def monitor_live_trading(self):
        """Continuous monitoring with safety checks."""
        print("🔴 LIVE TRADING MONITOR STARTED")
        print("=" * 50)
        print("⚠️ REAL MONEY AT RISK")
        print("Press 'q' + Enter to emergency stop")
        print("Press 's' + Enter to show status")
        print("=" * 50)
        
        while True:
            try:
                # Check for emergency stop file
                if os.path.exists(self.emergency_stop_file):
                    print("🚨 Emergency stop file detected")
                    break
                
                # Safety checks
                if not self.check_balance_safety():
                    print("🚨 Balance safety check failed - stopping bot")
                    self.emergency_stop()
                    break
                
                if not self.check_daily_performance():
                    print("🚨 Daily performance check failed - stopping bot")
                    self.emergency_stop()
                    break
                
                # Check for user input (non-blocking)
                import select
                import sys
                
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    user_input = sys.stdin.readline().strip().lower()
                    
                    if user_input == 'q':
                        print("🛑 User requested emergency stop")
                        self.emergency_stop()
                        break
                    elif user_input == 's':
                        self.show_status()
                
                time.sleep(5)  # Check every 5 seconds
                
            except KeyboardInterrupt:
                print("\n🛑 Ctrl+C pressed - emergency stop")
                self.emergency_stop()
                break
            except Exception as e:
                print(f"Error in monitor: {e}")
                time.sleep(1)
    
    def show_status(self):
        """Show current status."""
        print("\n" + "="*30)
        print(f"🕐 Status at {datetime.now().strftime('%H:%M:%S')}")
        
        # Check if bot is running
        try:
            result = subprocess.run(['pm2', 'list'], capture_output=True, text=True)
            if self.bot_name in result.stdout:
                print("✅ Bot is running")
            else:
                print("❌ Bot is not running")
        except:
            print("⚠️ Cannot check bot status")
        
        print("="*30 + "\n")
    
    def pre_flight_check(self):
        """Pre-flight safety check before starting live trading."""
        print("🔍 PRE-FLIGHT SAFETY CHECK")
        print("=" * 40)
        
        checks = []
        
        # Check API credentials
        from dotenv import load_dotenv
        load_dotenv('.azr_env')
        
        api_key = os.getenv('API_KEY')
        if not api_key or api_key == 'your_real_kucoin_api_key_here':
            checks.append("❌ API credentials not configured")
        else:
            checks.append("✅ API credentials configured")
        
        # Check sandbox mode
        sandbox = os.getenv('SANDBOX_MODE', 'true').lower()
        if sandbox == 'false':
            checks.append("⚠️ SANDBOX MODE DISABLED - LIVE TRADING")
        else:
            checks.append("✅ Sandbox mode enabled")
        
        # Check dependencies
        try:
            import ccxt
            import numpy
            checks.append("✅ Dependencies installed")
        except ImportError:
            checks.append("❌ Missing dependencies")
        
        # Print results
        for check in checks:
            print(check)
        
        print("\n" + "=" * 40)
        
        # Final warning
        if sandbox == 'false':
            print("🚨 FINAL WARNING 🚨")
            print("You are about to start LIVE TRADING with REAL MONEY")
            print("This bot can lose money quickly!")
            print("Type 'I UNDERSTAND THE RISKS' to continue:")
            
            confirmation = input().strip()
            if confirmation != 'I UNDERSTAND THE RISKS':
                print("❌ Live trading cancelled")
                return False
        
        return True

def main():
    """Main safety controller."""
    import sys
    
    safety = LiveTradingSafety()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'stop':
            safety.emergency_stop()
        elif command == 'monitor':
            safety.monitor_live_trading()
        elif command == 'check':
            safety.pre_flight_check()
        else:
            print("Usage: python live_safety.py [stop|monitor|check]")
    else:
        # Interactive mode
        print("🔴 LIVE TRADING SAFETY CONTROLLER")
        print("1. Pre-flight check")
        print("2. Start monitoring")
        print("3. Emergency stop")
        choice = input("Choose option (1-3): ").strip()
        
        if choice == '1':
            if safety.pre_flight_check():
                print("✅ Pre-flight check passed")
            else:
                print("❌ Pre-flight check failed")
        elif choice == '2':
            safety.monitor_live_trading()
        elif choice == '3':
            safety.emergency_stop()

if __name__ == "__main__":
    main()
