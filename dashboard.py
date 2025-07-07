#!/usr/bin/env python3
"""
Real-time Status Dashboard for AZR Trading Bot
Shows current status, logs, and controls
"""

import os
import time
import json
from datetime import datetime
import subprocess

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_latest_log():
    """Get the latest log entries."""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        return "No logs directory"
    
    today = datetime.now().strftime('%Y%m%d')
    log_file = f"{log_dir}/azr_bot_{today}.log"
    
    if not os.path.exists(log_file):
        return "No log file for today"
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return ''.join(lines[-10:])  # Last 10 lines
    except Exception as e:
        return f"Error reading log: {e}"

def check_bot_status():
    """Check if bot is running."""
    try:
        # Check if Python process is running the bot
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                               capture_output=True, text=True)
        if 'python.exe' in result.stdout:
            return "🟢 RUNNING"
        else:
            return "🔴 STOPPED"
    except:
        return "❓ UNKNOWN"

def get_performance_summary():
    """Get latest performance data."""
    data_dir = 'data'
    if not os.path.exists(data_dir):
        return "No performance data"
    
    perf_files = [f for f in os.listdir(data_dir) if f.startswith('performance_')]
    if not perf_files:
        return "No performance files"
    
    try:
        latest_file = sorted(perf_files)[-1]
        with open(os.path.join(data_dir, latest_file), 'r') as f:
            data = json.load(f)
            
        summary = f"Latest file: {latest_file}\n"
        if 'performance_log' in data and data['performance_log']:
            latest = data['performance_log'][-1]
            summary += f"Last cycle: {latest.get('trades_executed', 0)} trades\n"
            summary += f"Duration: {latest.get('duration', 0):.1f}s"
        
        return summary
    except Exception as e:
        return f"Error: {e}"

def show_dashboard():
    """Display the main dashboard."""
    clear_screen()
    
    print("🤖 KUCOIN AZR TRADING BOT - LIVE DASHBOARD")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Bot Status
    print(f"🔍 Bot Status: {check_bot_status()}")
    
    # Configuration Status
    from dotenv import load_dotenv
    load_dotenv('.azr_env')
    
    api_key = os.getenv('API_KEY', '')
    if api_key == 'your_real_kucoin_api_key_here':
        print("🔑 API Status: ⚠️ PLACEHOLDER CREDENTIALS")
        print("   Edit .azr_env with real KuCoin API keys to enable trading")
    else:
        print("🔑 API Status: ✅ CONFIGURED")
    
    sandbox = os.getenv('SANDBOX_MODE', 'true').lower()
    if sandbox == 'false':
        print("💰 Trading Mode: 🔴 LIVE TRADING (REAL MONEY)")
    else:
        print("💰 Trading Mode: 🟡 SANDBOX (SAFE)")
    
    print("-" * 60)
    
    # Recent Logs
    print("📋 RECENT LOGS:")
    recent_logs = get_latest_log()
    print(recent_logs)
    
    print("-" * 60)
    
    # Performance
    print("📊 PERFORMANCE:")
    perf_summary = get_performance_summary()
    print(perf_summary)
    
    print("=" * 60)
    print("🎮 CONTROLS: Press Ctrl+C to stop monitoring")
    print("   To stop bot: Close this window or press Ctrl+Break")

def main():
    """Main dashboard loop."""
    print("🚀 Starting AZR Trading Bot Dashboard...")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            show_dashboard()
            time.sleep(5)  # Update every 5 seconds
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")

if __name__ == "__main__":
    main()
