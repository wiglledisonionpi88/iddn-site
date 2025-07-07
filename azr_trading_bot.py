#!/usr/bin/env python3
"""
KuCoin AZR Self-Learning + Dust Profit Agent
Implementation for GitHub Issue #2

A cryptocurrency trading bot that performs automated trading on KuCoin
with self-learning capabilities and dust consolidation features.
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
        """Initialize the AZR Trading Bot with KuCoin exchange connection."""
        self.load_environment()
        self.setup_logging()
        self.setup_exchange()
        self.trading_data = {}
        self.performance_log = []
        self.dust_operations = []
        
    def load_environment(self):
        """Load environment variables from .azr_env file."""
        env_path = os.path.join(os.path.dirname(__file__), '.azr_env')
        load_dotenv(env_path)
        
        self.api_key = os.getenv('API_KEY')
        self.api_secret = os.getenv('API_SECRET')
        self.api_passphrase = os.getenv('API_PASSPHRASE')
        self.sandbox_mode = os.getenv('SANDBOX_MODE', 'true').lower() == 'true'
        
        # Trading configuration
        self.min_trade_amount = float(os.getenv('MIN_TRADE_AMOUNT', 10))
        self.max_dust_value = float(os.getenv('MAX_DUST_VALUE', 1.0))
        self.trading_pairs_count = int(os.getenv('TRADING_PAIRS_COUNT', 10))
        self.trading_cycle_minutes = int(os.getenv('TRADING_CYCLE_MINUTES', 5))
        self.momentum_threshold = float(os.getenv('MOMENTUM_THRESHOLD', 0.1))
        
        if not all([self.api_key, self.api_secret, self.api_passphrase]):
            raise ValueError("Missing required API credentials in .azr_env file")
    
    def setup_logging(self):
        """Setup comprehensive logging system."""
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
            self.exchange = ccxt.kucoin({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'password': self.api_passphrase,
                'sandbox': self.sandbox_mode,
                'enableRateLimit': True,
                'options': {
                    'adjustForTimeDifference': True,
                }
            })
            
            # Test connection
            self.exchange.load_markets()
            self.logger.info(f"Successfully connected to KuCoin {'Sandbox' if self.sandbox_mode else 'Live'}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to KuCoin: {e}")
            raise
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance for all currencies."""
        try:
            balance = self.exchange.fetch_balance()
            return {currency: info['free'] for currency, info in balance.items() 
                   if info['free'] > 0}
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return {}
    
    def identify_dust(self, balances: Dict[str, float]) -> List[Tuple[str, float]]:
        """Identify dust amounts (small coin amounts < $1)."""
        dust_coins = []
        
        try:
            for currency, amount in balances.items():
                if currency == 'USDT' or amount == 0:
                    continue
                
                # Get current price in USDT
                symbol = f"{currency}/USDT"
                if symbol in self.exchange.markets:
                    ticker = self.exchange.fetch_ticker(symbol)
                    value_usd = amount * ticker['last']
                    
                    if value_usd < self.max_dust_value and value_usd > 0.001:
                        dust_coins.append((currency, amount))
                        
        except Exception as e:
            self.logger.error(f"Error identifying dust: {e}")
        
        return dust_coins
    
    def consolidate_dust(self, dust_coins: List[Tuple[str, float]]) -> float:
        """Consolidate dust by selling to USDT."""
        total_dust_value = 0
        
        for currency, amount in dust_coins:
            try:
                symbol = f"{currency}/USDT"
                if symbol not in self.exchange.markets:
                    continue
                
                # Get minimum order size
                market = self.exchange.markets[symbol]
                min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
                
                if amount >= min_amount:
                    # Place market sell order
                    order = self.exchange.create_market_sell_order(symbol, amount)
                    
                    if order['status'] == 'closed':
                        usdt_received = order['cost']
                        total_dust_value += usdt_received
                        
                        dust_operation = {
                            'timestamp': datetime.now().isoformat(),
                            'currency': currency,
                            'amount': amount,
                            'usdt_received': usdt_received,
                            'order_id': order['id']
                        }
                        self.dust_operations.append(dust_operation)
                        
                        self.logger.info(f"Dust consolidated: {amount} {currency} -> {usdt_received:.4f} USDT")
                        
            except Exception as e:
                self.logger.error(f"Error consolidating dust for {currency}: {e}")
        
        return total_dust_value
    
    def get_top_trading_pairs(self) -> List[str]:
        """Get top trading pairs by 24h volume."""
        try:
            tickers = self.exchange.fetch_tickers()
            usdt_pairs = {symbol: ticker for symbol, ticker in tickers.items() 
                         if symbol.endswith('/USDT') and ticker['quoteVolume']}
            
            # Sort by 24h volume and get top N
            sorted_pairs = sorted(usdt_pairs.items(), 
                                key=lambda x: x[1]['quoteVolume'], reverse=True)
            
            top_pairs = [pair[0] for pair in sorted_pairs[:self.trading_pairs_count]]
            self.logger.info(f"Top trading pairs: {top_pairs}")
            
            return top_pairs
            
        except Exception as e:
            self.logger.error(f"Error getting top trading pairs: {e}")
            return []
    
    def analyze_price_data(self, symbol: str) -> Dict:
        """Analyze 1-minute OHLCV data for trend detection."""
        try:
            # Fetch 1-minute OHLCV data for last 100 candles
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1m', limit=100)
            
            if len(ohlcv) < 20:
                return {'signal': 'HOLD', 'confidence': 0, 'analysis': 'Insufficient data'}
            
            # Convert to numpy arrays for analysis
            closes = np.array([candle[4] for candle in ohlcv])
            volumes = np.array([candle[5] for candle in ohlcv])
            
            # Calculate percentage changes
            price_changes = np.diff(closes) / closes[:-1] * 100
            recent_change = price_changes[-1]
            avg_change = np.mean(price_changes[-10:])
            
            # Calculate volatility
            volatility = np.std(price_changes[-20:])
            
            # Volume analysis
            avg_volume = np.mean(volumes[-20:])
            recent_volume = volumes[-1]
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # Trading signal logic
            signal = 'HOLD'
            confidence = 0
            
            if recent_change > self.momentum_threshold and volume_ratio > 1.2:
                signal = 'BUY'
                confidence = min(abs(recent_change) * volume_ratio / 100, 0.8)
            elif recent_change < -self.momentum_threshold and volume_ratio > 1.2:
                signal = 'SELL'
                confidence = min(abs(recent_change) * volume_ratio / 100, 0.8)
            
            analysis = {
                'signal': signal,
                'confidence': confidence,
                'recent_change': recent_change,
                'avg_change': avg_change,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'analysis': f"Price change: {recent_change:.2f}%, Volume: {volume_ratio:.2f}x"
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing price data for {symbol}: {e}")
            return {'signal': 'HOLD', 'confidence': 0, 'analysis': f'Error: {e}'}
    
    def calculate_position_size(self, usdt_balance: float, pairs_count: int) -> float:
        """Calculate position size for equal distribution."""
        if pairs_count == 0:
            return 0
        
        # Reserve 10% for fees and slippage
        available_balance = usdt_balance * 0.9
        position_size = available_balance / pairs_count
        
        return max(position_size, self.min_trade_amount)
    
    def execute_trade(self, symbol: str, signal: str, position_size: float) -> Optional[Dict]:
        """Execute trading decision."""
        try:
            if signal == 'HOLD':
                return None
            
            market = self.exchange.markets[symbol]
            min_amount = market.get('limits', {}).get('cost', {}).get('min', self.min_trade_amount)
            
            if position_size < min_amount:
                self.logger.warning(f"Position size {position_size} below minimum {min_amount} for {symbol}")
                return None
            
            order = None
            if signal == 'BUY':
                order = self.exchange.create_market_buy_order(symbol, None, position_size)
            elif signal == 'SELL':
                # For sell, we need to check if we have the base currency
                base_currency = symbol.split('/')[0]
                balances = self.get_account_balance()
                
                if base_currency in balances and balances[base_currency] > 0:
                    amount = balances[base_currency]
                    order = self.exchange.create_market_sell_order(symbol, amount)
            
            if order and order.get('status') == 'closed':
                trade_log = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'signal': signal,
                    'order_id': order['id'],
                    'amount': order['amount'],
                    'cost': order['cost'],
                    'fee': order.get('fee', {}),
                    'price': order['price']
                }
                
                self.logger.info(f"Trade executed: {signal} {symbol} - Amount: {order['amount']}, Cost: {order['cost']}")
                return trade_log
            
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
        
        return None
    
    def save_performance_data(self):
        """Save performance data to JSON file."""
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'trading_data': self.trading_data,
            'dust_operations': self.dust_operations,
            'performance_log': self.performance_log
        }
        
        os.makedirs('data', exist_ok=True)
        filename = f"data/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(performance_data, f, indent=2)
            self.logger.info(f"Performance data saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving performance data: {e}")
    
    def run_trading_cycle(self):
        """Execute one complete trading cycle."""
        # Check for emergency stop
        if os.path.exists('EMERGENCY_STOP'):
            self.logger.critical("🚨 EMERGENCY STOP FILE DETECTED - SHUTTING DOWN")
            raise KeyboardInterrupt("Emergency stop requested")
        
        cycle_start = datetime.now()
        self.logger.info(f"Starting trading cycle at {cycle_start}")
        
        # Live trading warning
        if not self.sandbox_mode:
            self.logger.warning("🔴 LIVE TRADING MODE - REAL MONEY AT RISK")
        
        try:
            # 1. Get account balance
            balances = self.get_account_balance()
            usdt_balance = balances.get('USDT', 0)
            
            self.logger.info(f"Current USDT balance: {usdt_balance:.2f}")
            
            # 2. Consolidate dust
            dust_coins = self.identify_dust(balances)
            if dust_coins:
                dust_value = self.consolidate_dust(dust_coins)
                self.logger.info(f"Consolidated {len(dust_coins)} dust coins worth {dust_value:.4f} USDT")
                
                # Update USDT balance after dust consolidation
                balances = self.get_account_balance()
                usdt_balance = balances.get('USDT', 0)
            
            # 3. Get top trading pairs
            top_pairs = self.get_top_trading_pairs()
            
            if not top_pairs or usdt_balance < self.min_trade_amount:
                self.logger.warning(f"Insufficient balance ({usdt_balance}) or no trading pairs available")
                return
            
            # 4. Calculate position size
            position_size = self.calculate_position_size(usdt_balance, len(top_pairs))
            
            # 5. Analyze and trade each pair
            trades_executed = []
            
            for symbol in top_pairs:
                try:
                    # Analyze price data
                    analysis = self.analyze_price_data(symbol)
                    
                    # Store analysis data
                    self.trading_data[symbol] = {
                        'timestamp': datetime.now().isoformat(),
                        'analysis': analysis
                    }
                    
                    # Execute trade if signal confidence is high enough
                    if analysis['confidence'] > 0.3:
                        trade_result = self.execute_trade(symbol, analysis['signal'], position_size)
                        if trade_result:
                            trades_executed.append(trade_result)
                    
                    # Small delay to respect rate limits
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            # 6. Log cycle performance
            cycle_end = datetime.now()
            cycle_performance = {
                'cycle_start': cycle_start.isoformat(),
                'cycle_end': cycle_end.isoformat(),
                'duration': (cycle_end - cycle_start).total_seconds(),
                'pairs_analyzed': len(top_pairs),
                'trades_executed': len(trades_executed),
                'trades': trades_executed,
                'usdt_balance_start': usdt_balance,
                'dust_operations': len([op for op in self.dust_operations 
                                     if op['timestamp'].startswith(cycle_start.strftime('%Y-%m-%d'))])
            }
            
            self.performance_log.append(cycle_performance)
            
            self.logger.info(f"Trading cycle completed - Duration: {cycle_performance['duration']:.2f}s, "
                           f"Trades: {len(trades_executed)}, Pairs: {len(top_pairs)}")
            
            # 7. Save performance data every 10 cycles
            if len(self.performance_log) % 10 == 0:
                self.save_performance_data()
                
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
    
    def run(self):
        """Main bot loop - continuous operation with trading cycles."""
        self.logger.info("Starting AZR Trading Bot main loop")
        
        try:
            while True:
                # Run trading cycle
                self.run_trading_cycle()
                
                # Wait for next cycle
                wait_seconds = self.trading_cycle_minutes * 60
                self.logger.info(f"Waiting {wait_seconds} seconds until next cycle...")
                time.sleep(wait_seconds)
                
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
        except Exception as e:
            self.logger.error(f"Fatal error in main loop: {e}")
            raise
        finally:
            # Save final performance data
            self.save_performance_data()
            self.logger.info("AZR Trading Bot shutdown complete")

def main():
    """Main entry point for the AZR Trading Bot."""
    try:
        bot = AZRTradingBot()
        bot.run()
    except Exception as e:
        print(f"Failed to start AZR Trading Bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
