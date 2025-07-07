#!/usr/bin/env python3
"""
Simple Direct Bot Launcher
"""

import os
import sys
import subprocess

def main():
    print("🚀 Starting KuCoin AZR Trading Bot...")
    
    python_exe = "C:/Python313/python.exe"
    bot_script = "azr_trading_bot.py"
    
    # Change to bot directory
    os.chdir(r"c:\Users\Administrator\Desktop\C scripts\Documents\GitHub\iddn-site")
    
    # Start bot
    try:
        subprocess.run([python_exe, bot_script], check=True)
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
