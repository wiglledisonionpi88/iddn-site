# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.getcwd())

from azr_bot_clean import AZRTradingBot

print("=== QUICK BOT TEST ===")
try:
    bot = AZRTradingBot()
    print(f"Bot initialized in {'DEMO' if bot.demo_mode else 'LIVE'} mode")
    print(f"Exchange object: {bot.exchange}")
    print(f"Demo mode flag: {bot.demo_mode}")
    print("Running one trading cycle...")
    bot.run_trading_cycle()
    print("=== TEST COMPLETE ===")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
