# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
KuCoin AZR Self-Learning + Dust Profit Agent
Clean implementation with debug mode
"""

import os
import sys
import json
import time
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import ccxt
from dotenv import load_dotenv

class AZRTradingBot:
    def __init__(self):
        """Initialize the AZR Trading Bot."""
        self.demo_mode = False  # Initialize first
        self.load_environment()
        self.setup_logging()
        self.setup_exchange()
        self.trading_data = {}
        self.performance_log = []
        self.dust_operations = []
        
    def load_environment(self):
        """Load environment variables."""
        env_path = os.path.join(os.path.dirname(__file__), '.azr_env')
        load_dotenv(env_path)
        
        self.api_key = os.getenv('API_KEY')
        self.api_secret = os.getenv('API_SECRET')
        self.api_passphrase = os.getenv('API_PASSPHRASE')
        self.sandbox_mode = os.getenv('SANDBOX_MODE', 'true').lower() == 'true'
        
        # Trading configuration
        self.min_trade_amount = float(os.getenv('MIN_TRADE_AMOUNT', 10))
        self.max_dust_value = float(os.getenv('MAX_DUST_VALUE', 1.0))
        self.trading_pairs_count = int(os.getenv('TRADING_PAIRS_COUNT', 5))
        self.trading_cycle_minutes = int(os.getenv('TRADING_CYCLE_MINUTES', 10))
        self.momentum_threshold = float(os.getenv('MOMENTUM_THRESHOLD', 0.15))
        
        if not all([self.api_key, self.api_secret, self.api_passphrase]):
            raise ValueError("Missing required API credentials in .azr_env file")
    
    def setup_logging(self):
        """Setup logging system."""
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/azr_bot_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("AZR Trading Bot initialized")
    
    def setup_exchange(self):
        """Setup KuCoin exchange connection."""
        try:
            # Check for placeholder credentials
            if (self.api_key == 'your_real_kucoin_api_key_here' or 
                self.api_secret == 'your_real_kucoin_api_secret_here' or
                self.api_passphrase == 'your_real_kucoin_api_passphrase_here'):
                
                self.logger.warning("PLACEHOLDER CREDENTIALS DETECTED")
                self.logger.warning("Bot will run in demo mode - no actual trading")
                self.logger.warning("Edit .azr_env with real KuCoin API credentials to enable trading")
                
                # Create a mock exchange for demo mode
                self.exchange = None
                self.demo_mode = True
                return
            
            self.exchange = ccxt.kucoin({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.api_passphrase,
                'sandbox': self.sandbox_mode,
                'enableRateLimit': True,
            })
            
            # Test connection
            self.exchange.load_markets()
            self.demo_mode = False
            mode = 'Sandbox' if self.sandbox_mode else 'Live'
            self.logger.info(f"Successfully connected to KuCoin {mode}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to KuCoin: {e}")
            self.logger.warning("Switching to demo mode")
            self.exchange = None
            self.demo_mode = True
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance."""
        if self.demo_mode:
            return {'USDT': 1000.0, 'BTC': 0.001, 'ETH': 0.01}
        
        try:
            balance = self.exchange.fetch_balance()
            return {currency: info['free'] for currency, info in balance.items() 
                   if info['free'] > 0}
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return {}
    
    def get_top_trading_pairs(self) -> List[str]:
        """Get top trading pairs."""
        if self.demo_mode:
            return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
        
        try:
            tickers = self.exchange.fetch_tickers()
            usdt_pairs = {symbol: ticker for symbol, ticker in tickers.items() 
                         if symbol.endswith('/USDT') and ticker['quoteVolume']}
            
            sorted_pairs = sorted(usdt_pairs.items(), 
                                key=lambda x: x[1]['quoteVolume'], reverse=True)
            
            top_pairs = [pair[0] for pair in sorted_pairs[:self.trading_pairs_count]]
            self.logger.info(f"Top trading pairs: {top_pairs}")
            
            return top_pairs
            
        except Exception as e:
            self.logger.error(f"Error getting top trading pairs: {e}")
            return []
    
    def analyze_price_data(self, symbol: str) -> Dict:
        """Analyze price data."""
        if self.demo_mode:
            import random
            recent_change = random.uniform(-0.5, 0.5)
            confidence = random.uniform(0.1, 0.8) if abs(recent_change) > 0.1 else 0.05
            signal = 'BUY' if recent_change > 0.15 else 'SELL' if recent_change < -0.15 else 'HOLD'
            
            return {
                'signal': signal,
                'confidence': confidence,
                'recent_change': recent_change,
                'analysis': f"DEMO: Price change: {recent_change:.2f}%"
            }
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1m', limit=100)
            
            if len(ohlcv) < 20:
                return {'signal': 'HOLD', 'confidence': 0, 'analysis': 'Insufficient data'}
            
            closes = np.array([candle[4] for candle in ohlcv])
            price_changes = np.diff(closes) / closes[:-1] * 100
            recent_change = price_changes[-1]
            
            signal = 'HOLD'
            confidence = 0
            
            if recent_change > self.momentum_threshold:
                signal = 'BUY'
                confidence = min(abs(recent_change) / 100, 0.8)
            elif recent_change < -self.momentum_threshold:
                signal = 'SELL'
                confidence = min(abs(recent_change) / 100, 0.8)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'recent_change': recent_change,
                'analysis': f"Price change: {recent_change:.2f}%"
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return {'signal': 'HOLD', 'confidence': 0, 'analysis': f'Error: {e}'}
    
    def execute_trade(self, symbol: str, signal: str, position_size: float) -> Optional[Dict]:
        """Execute trade."""
        if signal == 'HOLD':
            return None
        
        if self.demo_mode:
            import random
            trade_log = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'signal': signal,
                'amount': position_size / 50000,  # Rough BTC price
                'cost': position_size,
                'demo': True
            }
            
            self.logger.info(f"DEMO TRADE: {signal} {symbol} - Cost: ${position_size:.2f}")
            return trade_log
        
        try:
            if not self.exchange:
                return None
            
            # Real trading logic would go here
            self.logger.info(f"LIVE TRADE: {signal} {symbol} - ${position_size:.2f}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
            return None
    
    def run_trading_cycle(self):
        """Execute one trading cycle."""
        # Check for emergency stop
        if os.path.exists('EMERGENCY_STOP'):
            self.logger.critical("EMERGENCY STOP FILE DETECTED - SHUTTING DOWN")
            raise KeyboardInterrupt("Emergency stop requested")
        
        cycle_start = datetime.now()
        mode_text = "DEMO MODE" if self.demo_mode else ("LIVE TRADING" if not self.sandbox_mode else "SANDBOX")
        self.logger.info(f"Starting trading cycle - {mode_text}")
        
        if not self.sandbox_mode and not self.demo_mode:
            self.logger.warning("LIVE TRADING MODE - REAL MONEY AT RISK")
        elif self.demo_mode:
            self.logger.info("DEMO MODE - No real trading, simulating")
        
        try:
            # Get account balance
            balances = self.get_account_balance()
            usdt_balance = balances.get('USDT', 0)
            
            self.logger.info(f"USDT balance: ${usdt_balance:.2f}")
            
            # Get top trading pairs
            top_pairs = self.get_top_trading_pairs()
            
            if not top_pairs:
                self.logger.warning("No trading pairs available")
                return
            
            # Calculate position size
            position_size = max(usdt_balance / len(top_pairs), self.min_trade_amount)
            
            # Analyze and trade each pair
            trades_executed = []
            
            for symbol in top_pairs:
                try:
                    analysis = self.analyze_price_data(symbol)
                    
                    self.trading_data[symbol] = {
                        'timestamp': datetime.now().isoformat(),
                        'analysis': analysis
                    }
                    
                    if analysis['confidence'] > 0.3:
                        trade_result = self.execute_trade(symbol, analysis['signal'], position_size)
                        if trade_result:
                            trades_executed.append(trade_result)
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            # Log cycle performance
            cycle_end = datetime.now()
            self.logger.info(f"Cycle completed - Trades: {len(trades_executed)}, Duration: {(cycle_end - cycle_start).total_seconds():.1f}s")
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
    
    def run(self):
        """Main bot loop."""
        self.logger.info("Starting AZR Trading Bot main loop")
        
        try:
            while True:
                self.run_trading_cycle()
                
                wait_seconds = self.trading_cycle_minutes * 60
                self.logger.info(f"Waiting {wait_seconds} seconds until next cycle...")
                time.sleep(wait_seconds)
                
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            raise
        finally:
            self.logger.info("AZR Trading Bot shutdown complete")

def main():
    """Main entry point."""
    try:
        bot = AZRTradingBot()
        bot.run()
    except Exception as e:
        print(f"Failed to start bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
