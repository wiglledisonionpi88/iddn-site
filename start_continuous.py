# -*- coding: utf-8 -*-
"""
Continuous Bot Runner
"""
import sys
import os
sys.path.append(os.getcwd())

print("🚀 STARTING KUCOIN AZR TRADING BOT - CONTINUOUS MODE")
print("=" * 60)

try:
    from azr_bot_clean import AZRTradingBot
    bot = AZRTradingBot()
    
    mode = "DEMO" if bot.demo_mode else "LIVE"
    print(f"🤖 Bot Mode: {mode}")
    print(f"💰 Starting Balance: $1000 (demo)" if bot.demo_mode else "💰 Checking live balance...")
    print("🔄 Running continuous trading cycles...")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 60)
    
    bot.run()
    
except KeyboardInterrupt:
    print("\n🛑 Bot stopped by user")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
