#!/usr/bin/env python3
"""
AZR Trading Bot Monitoring Script
Real-time monitoring and alerts
"""

import json
import time
import os
from datetime import datetime, timedelta
import subprocess

class BotMonitor:
    def __init__(self):
        self.bot_name = "azr-trading-bot"
        self.alert_thresholds = {
            'memory_mb': 500,
            'restart_count': 5,
            'error_rate': 0.1
        }
    
    def get_pm2_status(self):
        """Get PM2 process status."""
        try:
            result = subprocess.run(['pm2', 'jlist'], capture_output=True, text=True)
            if result.returncode == 0:
                processes = json.loads(result.stdout)
                for proc in processes:
                    if proc['name'] == self.bot_name:
                        return proc
            return None
        except Exception as e:
            print(f"Error getting PM2 status: {e}")
            return None
    
    def check_log_files(self):
        """Check recent log files for errors."""
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            return {'error_count': 0, 'total_lines': 0}
        
        today = datetime.now().strftime('%Y%m%d')
        log_file = f"{log_dir}/azr_bot_{today}.log"
        
        if not os.path.exists(log_file):
            return {'error_count': 0, 'total_lines': 0}
        
        error_count = 0
        total_lines = 0
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    total_lines += 1
                    if 'ERROR' in line or 'CRITICAL' in line:
                        error_count += 1
        except Exception as e:
            print(f"Error reading log file: {e}")
        
        return {'error_count': error_count, 'total_lines': total_lines}
    
    def check_performance_data(self):
        """Check latest performance data."""
        data_dir = 'data'
        if not os.path.exists(data_dir):
            return None
        
        # Get most recent performance file
        perf_files = [f for f in os.listdir(data_dir) if f.startswith('performance_')]
        if not perf_files:
            return None
        
        latest_file = sorted(perf_files)[-1]
        file_path = os.path.join(data_dir, latest_file)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error reading performance data: {e}")
            return None
    
    def display_status(self):
        """Display comprehensive bot status."""
        print("\n🤖 AZR Trading Bot Status")
        print("=" * 60)
        
        # PM2 Status
        pm2_status = self.get_pm2_status()
        if pm2_status:
            status = pm2_status['pm2_env']['status']
            memory = pm2_status['monit']['memory'] / 1024 / 1024  # Convert to MB
            cpu = pm2_status['monit']['cpu']
            restarts = pm2_status['pm2_env']['restart_time']
            uptime = pm2_status['pm2_env']['pm_uptime']
            
            print(f"📊 Process Status: {status}")
            print(f"💾 Memory Usage: {memory:.1f} MB")
            print(f"⚡ CPU Usage: {cpu}%")
            print(f"🔄 Restarts: {restarts}")
            
            if uptime:
                uptime_hours = (time.time() * 1000 - uptime) / (1000 * 60 * 60)
                print(f"⏱️  Uptime: {uptime_hours:.1f} hours")
            
            # Memory alert
            if memory > self.alert_thresholds['memory_mb']:
                print(f"⚠️  HIGH MEMORY USAGE: {memory:.1f} MB")
            
            # Restart alert
            if restarts > self.alert_thresholds['restart_count']:
                print(f"⚠️  HIGH RESTART COUNT: {restarts}")
        else:
            print("❌ Bot not running or PM2 not available")
        
        print("-" * 60)
        
        # Log Analysis
        log_data = self.check_log_files()
        if log_data['total_lines'] > 0:
            error_rate = log_data['error_count'] / log_data['total_lines']
            print(f"📋 Log Analysis:")
            print(f"   Total log lines: {log_data['total_lines']}")
            print(f"   Error count: {log_data['error_count']}")
            print(f"   Error rate: {error_rate:.2%}")
            
            if error_rate > self.alert_thresholds['error_rate']:
                print(f"⚠️  HIGH ERROR RATE: {error_rate:.2%}")
        else:
            print("📋 No log data available")
        
        print("-" * 60)
        
        # Performance Data
        perf_data = self.check_performance_data()
        if perf_data:
            if 'performance_log' in perf_data and perf_data['performance_log']:
                latest_cycle = perf_data['performance_log'][-1]
                print(f"📈 Latest Trading Cycle:")
                print(f"   Duration: {latest_cycle.get('duration', 0):.1f}s")
                print(f"   Pairs analyzed: {latest_cycle.get('pairs_analyzed', 0)}")
                print(f"   Trades executed: {latest_cycle.get('trades_executed', 0)}")
                
                if 'usdt_balance_start' in latest_cycle:
                    print(f"   USDT balance: ${latest_cycle['usdt_balance_start']:.2f}")
        else:
            print("📈 No performance data available")
        
        print("=" * 60)
    
    def monitor_loop(self, interval=30):
        """Continuous monitoring loop."""
        print(f"🔍 Starting bot monitoring (refresh every {interval}s)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                self.display_status()
                print(f"\n⏰ Next update in {interval} seconds...")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")

def main():
    """Main monitoring function."""
    import sys
    
    monitor = BotMonitor()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'loop':
        # Continuous monitoring
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        monitor.monitor_loop(interval)
    else:
        # Single status check
        monitor.display_status()

if __name__ == "__main__":
    main()
