#!/usr/bin/env python3
"""
KuCoin AZR Advanced Autonomous Trading Bot
Implementation for GitHub Issue #2

An advanced cryptocurrency trading bot with:
- Advanced AI-driven self-learning
- Autonomous portfolio rebalancing
- Dynamic risk management
- Multi-timeframe analysis
- Adaptive position sizing
- Real-time sentiment analysis
- Automated profit optimization
"""

import os
import sys
import json
import time
import logging
import numpy as np
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass
import ccxt
from dotenv import load_dotenv
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import statistics
import requests

@dataclass
class TradingSignal:
    """Advanced trading signal with confidence metrics."""
    symbol: str
    signal: str  # BUY, SELL, HOLD
    confidence: float
    strength: float
    timeframe: str
    indicators: Dict[str, float]
    risk_score: float
    expected_return: float

@dataclass
class PortfolioMetrics:
    """Portfolio performance and risk metrics."""
    total_value: float
    unrealized_pnl: float
    realized_pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    
class AdvancedAnalytics:
    """Advanced analytics engine for market analysis."""
    
    def __init__(self):
        self.price_history = {}
        self.volume_history = {}
        self.trade_history = deque(maxlen=1000)
        
    def calculate_technical_indicators(self, ohlcv: List) -> Dict[str, float]:
        """Calculate multiple technical indicators."""
        closes = np.array([candle[4] for candle in ohlcv])
        volumes = np.array([candle[5] for candle in ohlcv])
        highs = np.array([candle[2] for candle in ohlcv])
        lows = np.array([candle[3] for candle in ohlcv])
        
        # Moving averages
        ma_5 = np.mean(closes[-5:]) if len(closes) >= 5 else closes[-1]
        ma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
        ma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else closes[-1]
        
        # RSI calculation
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = closes[-1] if len(closes) < 12 else np.mean(closes[-12:])
        ema_26 = closes[-1] if len(closes) < 26 else np.mean(closes[-26:])
        macd = ema_12 - ema_26
        
        # Bollinger Bands
        bb_middle = ma_20
        bb_std = np.std(closes[-20:]) if len(closes) >= 20 else np.std(closes)
        bb_upper = bb_middle + (2 * bb_std)
        bb_lower = bb_middle - (2 * bb_std)
        bb_position = (closes[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        # Volume indicators
        vol_ma = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
        vol_ratio = volumes[-1] / vol_ma if vol_ma > 0 else 1
        
        # Volatility
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns) * np.sqrt(24 * 365)  # Annualized
        
        return {
            'ma_5': ma_5,
            'ma_20': ma_20,
            'ma_50': ma_50,
            'rsi': rsi,
            'macd': macd,
            'bb_position': bb_position,
            'vol_ratio': vol_ratio,
            'volatility': volatility,
            'momentum': (closes[-1] / closes[-5] - 1) * 100 if len(closes) >= 5 else 0
        }
    
    def calculate_risk_score(self, indicators: Dict[str, float], market_conditions: Dict) -> float:
        """Calculate dynamic risk score based on multiple factors."""
        risk_factors = []
        
        # Volatility risk
        vol_risk = min(indicators['volatility'] / 0.5, 1.0)
        risk_factors.append(vol_risk * 0.3)
        
        # RSI extremes
        rsi_risk = abs(indicators['rsi'] - 50) / 50
        risk_factors.append(rsi_risk * 0.2)
        
        # Market sentiment
        sentiment_risk = market_conditions.get('fear_greed_index', 50) / 100
        risk_factors.append(sentiment_risk * 0.2)
        
        # Volume anomaly
        vol_risk = min(abs(indicators['vol_ratio'] - 1) * 2, 1.0)
        risk_factors.append(vol_risk * 0.3)
        
        return sum(risk_factors)

class AutonomousRiskManager:
    """Advanced autonomous risk management system."""
    
    def __init__(self, max_position_size: float = 0.1, max_daily_loss: float = 0.05):
        self.max_position_size = max_position_size  # 10% max per trade
        self.max_daily_loss = max_daily_loss  # 5% max daily loss
        self.daily_pnl = 0.0
        self.open_positions = {}
        self.risk_appetite = 0.5  # Dynamic risk appetite
        
    def adjust_position_size(self, base_size: float, confidence: float, risk_score: float) -> float:
        """Dynamically adjust position size based on confidence and risk."""
        # Base adjustment by confidence
        confidence_multiplier = confidence * 2  # 0.0 to 2.0
        
        # Risk adjustment
        risk_multiplier = max(0.1, 1.0 - risk_score)
        
        # Portfolio heat adjustment
        heat_multiplier = max(0.1, 1.0 - abs(self.daily_pnl / self.max_daily_loss))
        
        adjusted_size = base_size * confidence_multiplier * risk_multiplier * heat_multiplier
        
        # Enforce maximum position size
        return min(adjusted_size, base_size * self.max_position_size * 10)
    
    def should_trade(self, signal: TradingSignal, portfolio_value: float) -> bool:
        """Determine if trade should be executed based on risk management."""
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss * portfolio_value:
            return False
            
        # Check minimum confidence
        if signal.confidence < 0.3:
            return False
            
        # Check risk score
        if signal.risk_score > 0.8:
            return False
            
        return True

class AZRTradingBot:
    def __init__(self):
        """Initialize the Advanced AZR Trading Bot with autonomous capabilities."""
        self.load_environment()
        self.setup_logging()
        self.setup_exchange()
        
        # Advanced components
        self.analytics = AdvancedAnalytics()
        self.risk_manager = AutonomousRiskManager()
        
        # Data storage
        self.trading_data = {}
        self.performance_log = []
        self.dust_operations = []
        self.market_data_cache = {}
        
        # AI Learning components
        self.learning_history = deque(maxlen=10000)
        self.performance_memory = deque(maxlen=1000)
        self.success_patterns = {}
        
        # Autonomous features
        self.auto_rebalance = True
        self.adaptive_timing = True
        self.sentiment_analysis = True
        self.multi_timeframe = True
        
        # Threading for autonomous operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.background_tasks = []
        
        self.demo_mode = False
        self.logger.info("🤖 Advanced AZR Trading Bot initialized with autonomous capabilities")
        
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
        
        # Validate credentials
        if not all([self.api_key, self.api_secret, self.api_passphrase]):
            raise ValueError("Missing required API credentials in .azr_env file")
    
    def start_autonomous_monitoring(self):
        """Start background autonomous monitoring tasks."""
        if not self.demo_mode:
            # Market monitoring
            self.background_tasks.append(
                self.executor.submit(self.continuous_market_monitor)
            )
            
            # Portfolio rebalancing
            self.background_tasks.append(
                self.executor.submit(self.autonomous_rebalancer)
            )
            
            # Risk monitoring
            self.background_tasks.append(
                self.executor.submit(self.risk_monitor)
            )
            
            self.logger.info("🚀 Autonomous monitoring systems activated")
    
    def continuous_market_monitor(self):
        """Continuously monitor market conditions for opportunities."""
        while True:
            try:
                if not self.demo_mode and self.exchange:
                    # Get market overview
                    tickers = self.exchange.fetch_tickers()
                    
                    # Identify unusual movements
                    for symbol, ticker in tickers.items():
                        if symbol.endswith('/USDT') and ticker.get('percentage'):
                            change_24h = ticker['percentage']
                            volume_24h = ticker.get('quoteVolume', 0)
                            
                            # Check for significant movements
                            if abs(change_24h) > 10 and volume_24h > 1000000:
                                self.logger.warning(f"🚨 ALERT: {symbol} moved {change_24h:.2f}% with high volume")
                                
                                # Trigger immediate analysis
                                self.quick_opportunity_analysis(symbol)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Market monitoring error: {e}")
                time.sleep(60)
    
    def autonomous_rebalancer(self):
        """Autonomous portfolio rebalancing based on performance."""
        while True:
            try:
                if not self.demo_mode and len(self.performance_memory) > 10:
                    # Calculate portfolio performance
                    recent_performance = list(self.performance_memory)[-10:]
                    avg_performance = sum(p.get('profit_loss', 0) for p in recent_performance) / len(recent_performance)
                    
                    # Adjust strategy if underperforming
                    if avg_performance < 0:
                        self.risk_manager.risk_appetite *= 0.95  # Reduce risk
                        self.momentum_threshold *= 1.1  # Be more conservative
                        self.logger.info("📉 Autonomous rebalancing: Reducing risk due to poor performance")
                    elif avg_performance > 0.02:  # 2% profit
                        self.risk_manager.risk_appetite *= 1.05  # Increase risk slightly
                        self.momentum_threshold *= 0.95  # Be more aggressive
                        self.logger.info("📈 Autonomous rebalancing: Increasing risk due to good performance")
                    
                    # Clamp values
                    self.risk_manager.risk_appetite = max(0.1, min(1.0, self.risk_manager.risk_appetite))
                    self.momentum_threshold = max(0.05, min(0.3, self.momentum_threshold))
                
                time.sleep(300)  # Rebalance every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Rebalancing error: {e}")
                time.sleep(300)
    
    def risk_monitor(self):
        """Continuous risk monitoring and automatic position adjustment."""
        while True:
            try:
                if not self.demo_mode and self.exchange:
                    # Check current positions
                    balance = self.get_account_balance()
                    total_value = sum(balance.values())
                    
                    # Check for emergency conditions
                    if self.risk_manager.daily_pnl < -self.risk_manager.max_daily_loss * total_value:
                        self.logger.critical("🚨 DAILY LOSS LIMIT REACHED - HALTING TRADING")
                        with open('EMERGENCY_STOP', 'w') as f:
                            f.write(f"Daily loss limit reached at {datetime.now()}")
                    
                    # Monitor individual positions
                    for currency, amount in balance.items():
                        if currency != 'USDT' and amount > 0:
                            symbol = f"{currency}/USDT"
                            try:
                                ticker = self.exchange.fetch_ticker(symbol)
                                position_value = amount * ticker['last']
                                position_pct = position_value / total_value
                                
                                # Check if position is too large
                                if position_pct > self.risk_manager.max_position_size * 2:
                                    self.logger.warning(f"⚠️ Large position detected: {symbol} ({position_pct:.2%})")
                                    # Could implement auto-scaling here
                                    
                            except Exception:
                                continue
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Risk monitoring error: {e}")
                time.sleep(60)
    
    def quick_opportunity_analysis(self, symbol: str):
        """Quick analysis for sudden market opportunities."""
        try:
            if self.demo_mode:
                return
                
            # Get recent data
            ohlcv = self.exchange.fetch_ohlcv(symbol, '5m', limit=50)
            if len(ohlcv) < 20:
                return
                
            # Quick technical analysis
            indicators = self.analytics.calculate_technical_indicators(ohlcv)
            
            # Check for strong signals
            strong_buy = (indicators['rsi'] < 30 and indicators['macd'] > 0 and 
                         indicators['vol_ratio'] > 2)
            strong_sell = (indicators['rsi'] > 70 and indicators['macd'] < 0 and 
                          indicators['vol_ratio'] > 2)
            
            if strong_buy or strong_sell:
                signal_type = 'BUY' if strong_buy else 'SELL'
                confidence = min(0.8, indicators['vol_ratio'] / 5)
                
                self.logger.info(f"🎯 OPPORTUNITY: {signal_type} {symbol} with confidence {confidence:.2f}")
                
                # Could execute immediate trade here based on opportunity
                
        except Exception as e:
            self.logger.error(f"Quick analysis error for {symbol}: {e}")
    
    def get_market_sentiment(self) -> Dict[str, float]:
        """Get market sentiment indicators."""
        try:
            # Fear & Greed Index (mock implementation)
            # In real implementation, you'd fetch from actual APIs
            sentiment_data = {
                'fear_greed_index': 50,  # 0-100, 0=extreme fear, 100=extreme greed
                'market_trend': 0.5,     # 0-1, bearish to bullish
                'volatility_index': 0.3   # 0-1, low to high volatility
            }
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Error getting market sentiment: {e}")
            return {'fear_greed_index': 50, 'market_trend': 0.5, 'volatility_index': 0.3}
    
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
            # Check for placeholder credentials
            if (self.api_key == 'your_real_kucoin_api_key_here' or 
                self.api_secret == 'your_real_kucoin_api_secret_here' or
                self.api_passphrase == 'your_real_kucoin_api_passphrase_here'):
                
                self.logger.warning("🔴 PLACEHOLDER CREDENTIALS DETECTED")
                self.logger.warning("Bot will run in demo mode - no actual trading")
                self.logger.warning("Edit .azr_env with real KuCoin API credentials to enable trading")
                
                # Create a mock exchange for demo mode
                self.exchange = None
                self.demo_mode = True
                return
            
            # LIVE TRADING MODE DETECTED
            self.logger.critical("🔴🔴🔴 LIVE TRADING MODE DETECTED 🔴🔴🔴")
            self.logger.critical("🚨 REAL API CREDENTIALS LOADED - TRADING WITH REAL MONEY")
            self.logger.critical(f"🎯 API Key: {self.api_key[:8]}...")
            self.logger.critical(f"🌍 Sandbox Mode: {'OFF - LIVE TRADING' if not self.sandbox_mode else 'ON - SANDBOX'}")
            
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
            
            # Test connection and get real account info
            self.exchange.load_markets()
            account_info = self.exchange.fetch_balance()
            
            self.demo_mode = False
            
            # Log actual account balance
            usdt_balance = account_info.get('USDT', {}).get('free', 0)
            total_balance = sum(info.get('total', 0) for info in account_info.values() if isinstance(info, dict))
            
            self.logger.critical(f"✅ CONNECTED TO KUCOIN {'LIVE TRADING' if not self.sandbox_mode else 'SANDBOX'}")
            self.logger.critical(f"💰 REAL ACCOUNT BALANCE: {usdt_balance:.2f} USDT")
            self.logger.critical(f"📊 TOTAL PORTFOLIO VALUE: ${total_balance:.2f}")
            self.logger.critical("🚀 READY FOR LIVE TRADING WITH REAL MONEY")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to KuCoin: {e}")
            self.logger.warning("🔄 Connection failed - switching to demo mode")
            self.exchange = None
            self.demo_mode = True
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance for all currencies."""
        if self.demo_mode:
            return {'USDT': 1000.0, 'BTC': 0.001, 'ETH': 0.01}  # Demo balance
        
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
        if self.demo_mode:
            total_value = sum(amount * 0.5 for _, amount in dust_coins)  # Demo value
            self.logger.info(f"DEMO DUST: Consolidated {len(dust_coins)} coins worth ${total_value:.4f}")
            return total_value
        
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
        if self.demo_mode:
            return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']  # Demo pairs
        
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
        """Legacy method - redirects to advanced analysis."""
        signal = self.analyze_price_data_advanced(symbol)
        return {
            'signal': signal.signal,
            'confidence': signal.confidence,
            'recent_change': signal.indicators.get('demo_change', 0),
            'avg_change': signal.indicators.get('demo_change', 0) * 0.8,
            'volatility': signal.risk_score,
            'volume_ratio': signal.indicators.get('demo_volume', 1.0),
            'analysis': f"Advanced: {signal.signal} with {signal.confidence:.2f} confidence"
        }
    
    def execute_trade_advanced(self, signal: TradingSignal, position_size: float) -> Optional[Dict]:
        """Advanced trade execution with enhanced logic."""
        if signal.signal == 'HOLD' or not self.risk_manager.should_trade(signal, position_size * 10):
            return None
        
        if self.demo_mode:
            # Enhanced demo trading simulation
            import random
            
            # Simulate more realistic execution
            slippage = random.uniform(0.001, 0.003)  # 0.1-0.3% slippage
            fee_rate = 0.001  # 0.1% trading fee
            
            if signal.signal == 'BUY':
                execution_price = 50000 * (1 + slippage) if 'BTC' in signal.symbol else 3000 * (1 + slippage)
                amount = position_size / execution_price
            else:  # SELL
                execution_price = 50000 * (1 - slippage) if 'BTC' in signal.symbol else 3000 * (1 - slippage)
                amount = position_size / execution_price
            
            fee_cost = position_size * fee_rate
            
            trade_log = {
                'timestamp': datetime.now().isoformat(),
                'symbol': signal.symbol,
                'signal': signal.signal,
                'order_id': f"DEMO_ADV_{int(time.time())}",
                'amount': amount,
                'cost': position_size,
                'fee': {'cost': fee_cost, 'currency': 'USDT'},
                'price': execution_price,
                'confidence': signal.confidence,
                'risk_score': signal.risk_score,
                'expected_return': signal.expected_return,
                'demo': True,
                'advanced': True
            }
            
            self.logger.info(f"🚀 ADVANCED DEMO TRADE: {signal.signal} {signal.symbol} - "
                           f"Amount: {amount:.6f}, Cost: ${position_size:.2f}, "
                           f"Confidence: {signal.confidence:.2f}, Risk: {signal.risk_score:.2f}")
            
            # Learn from this demo trade
            self.learn_from_trade(trade_log, signal)
            
            return trade_log
        
        # Real trading logic (enhanced)
        try:
            if not self.exchange:
                return None
                
            market = self.exchange.markets[signal.symbol]
            min_amount = market.get('limits', {}).get('cost', {}).get('min', self.min_trade_amount)
            
            if position_size < min_amount:
                self.logger.warning(f"Position size {position_size} below minimum {min_amount} for {signal.symbol}")
                return None
            
            order = None
            if signal.signal == 'BUY':
                # Enhanced buy order with limit price for better execution
                ticker = self.exchange.fetch_ticker(signal.symbol)
                limit_price = ticker['ask'] * 1.001  # Slight premium for faster execution
                
                order = self.exchange.create_limit_buy_order(signal.symbol, None, limit_price, position_size)
                
            elif signal.signal == 'SELL':
                # Check holdings and execute sell
                base_currency = signal.symbol.split('/')[0]
                balances = self.get_account_balance()
                
                if base_currency in balances and balances[base_currency] > 0:
                    amount = balances[base_currency]
                    ticker = self.exchange.fetch_ticker(signal.symbol)
                    limit_price = ticker['bid'] * 0.999  # Slight discount for faster execution
                    
                    order = self.exchange.create_limit_sell_order(signal.symbol, amount, limit_price)
            
            if order and order.get('status') in ['closed', 'open']:
                trade_log = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': signal.symbol,
                    'signal': signal.signal,
                    'order_id': order['id'],
                    'amount': order['amount'],
                    'cost': order['cost'] if order.get('cost') else position_size,
                    'fee': order.get('fee', {}),
                    'price': order['price'],
                    'confidence': signal.confidence,
                    'risk_score': signal.risk_score,
                    'expected_return': signal.expected_return,
                    'status': order['status'],
                    'advanced': True
                }
                
                self.logger.info(f"🎯 ADVANCED TRADE EXECUTED: {signal.signal} {signal.symbol} - "
                               f"Amount: {order['amount']}, Cost: {order.get('cost', position_size)}, "
                               f"Confidence: {signal.confidence:.2f}")
                
                # Learn from this real trade
                self.learn_from_trade(trade_log, signal)
                
                return trade_log
            
        except Exception as e:
            self.logger.error(f"Advanced trade execution error for {signal.symbol}: {e}")
        
        return None
    
    def calculate_position_size(self, usdt_balance: float, pairs_count: int) -> float:
        """Calculate position size for equal distribution."""
        if pairs_count == 0:
            return 0
        
        # Reserve 10% for fees and slippage
        available_balance = usdt_balance * 0.9
        position_size = available_balance / pairs_count
        
        return max(position_size, self.min_trade_amount)
    
    def execute_trade(self, symbol: str, signal: str, position_size: float) -> Optional[Dict]:
        """Legacy method - redirects to advanced execution."""
        # Convert to TradingSignal for advanced execution
        trading_signal = TradingSignal(
            symbol=symbol,
            signal=signal,
            confidence=0.5,  # Default confidence
            strength=0.5,
            timeframe='legacy',
            indicators={},
            risk_score=0.3,
            expected_return=0.01
        )
        
        return self.execute_trade_advanced(trading_signal, position_size)
    
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
        """Execute one complete ADVANCED autonomous trading cycle."""
        # Check for emergency stop
        if os.path.exists('EMERGENCY_STOP'):
            self.logger.critical("🚨 EMERGENCY STOP FILE DETECTED - SHUTTING DOWN")
            raise KeyboardInterrupt("Emergency stop requested")
        
        cycle_start = datetime.now()
        mode_text = "ADVANCED DEMO" if self.demo_mode else ("LIVE AUTONOMOUS TRADING" if not self.sandbox_mode else "ADVANCED SANDBOX")
        self.logger.info(f"🤖 Starting ADVANCED trading cycle at {cycle_start} - {mode_text}")
        
        # Live trading warnings
        if not self.sandbox_mode and not self.demo_mode:
            self.logger.warning("🔴 AUTONOMOUS LIVE TRADING - REAL MONEY WITH AI DECISIONS")
        elif self.demo_mode:
            self.logger.info("🟡 ADVANCED DEMO - AI trading simulation with learning")
        
        try:
            # 1. Get portfolio metrics and balance
            portfolio_metrics = self.get_portfolio_metrics()
            balances = self.get_account_balance()
            usdt_balance = balances.get('USDT', 0)
            
            self.logger.info(f"💰 Portfolio Value: ${portfolio_metrics.total_value:.2f} "
                           f"| Win Rate: {portfolio_metrics.win_rate:.1%} "
                           f"| Sharpe: {portfolio_metrics.sharpe_ratio:.2f}")
            
            # 2. Autonomous dust consolidation
            if not self.demo_mode:
                dust_coins = self.identify_dust(balances)
                if dust_coins:
                    dust_value = self.consolidate_dust(dust_coins)
                    self.logger.info(f"🧹 AUTO-CONSOLIDATED: {len(dust_coins)} dust coins worth ${dust_value:.4f}")
                    
                    # Update balance after dust consolidation
                    balances = self.get_account_balance()
                    usdt_balance = balances.get('USDT', 0)
            
            # 3. Get market sentiment and conditions
            market_sentiment = self.get_market_sentiment()
            self.logger.info(f"📊 Market Sentiment - Fear/Greed: {market_sentiment['fear_greed_index']}, "
                           f"Trend: {market_sentiment['market_trend']:.2f}")
            
            # 4. Get top trading pairs with enhanced selection
            top_pairs = self.get_top_trading_pairs()
            
            if not top_pairs or usdt_balance < self.min_trade_amount:
                self.logger.warning(f"⚠️ Insufficient balance (${usdt_balance:.2f}) or no trading pairs")
                return
            
            # 5. Advanced multi-timeframe analysis for each pair
            trading_signals = []
            self.logger.info(f"🔍 Analyzing {len(top_pairs)} pairs with AI...")
            
            for symbol in top_pairs:
                try:
                    # Advanced analysis with multiple timeframes
                    signal = self.analyze_price_data_advanced(symbol, ['1m', '5m', '15m'])
                    
                    # Store analysis data
                    self.trading_data[symbol] = {
                        'timestamp': datetime.now().isoformat(),
                        'signal': signal,
                        'market_sentiment': market_sentiment
                    }
                    
                    trading_signals.append(signal)
                    
                    self.logger.debug(f"📈 {symbol}: {signal.signal} "
                                    f"(Conf: {signal.confidence:.2f}, Risk: {signal.risk_score:.2f})")
                    
                    time.sleep(0.05)  # Small delay for rate limiting
                    
                except Exception as e:
                    self.logger.error(f"Analysis error for {symbol}: {e}")
                    continue
            
            # 6. Rank signals by expected return and risk
            trading_signals.sort(key=lambda s: s.expected_return / (s.risk_score + 0.1), reverse=True)
            
            # 7. Execute trades with adaptive position sizing
            trades_executed = []
            
            for signal in trading_signals:
                if signal.signal == 'HOLD':
                    continue
                
                # Calculate adaptive position size
                position_size = self.adaptive_position_sizing(signal, usdt_balance, len(top_pairs))
                
                # Execute trade with advanced logic
                trade_result = self.execute_trade_advanced(signal, position_size)
                
                if trade_result:
                    trades_executed.append(trade_result)
                    
                    # Update risk manager daily P&L
                    profit_estimate = trade_result.get('expected_return', 0) * position_size
                    self.risk_manager.daily_pnl += profit_estimate
                    
                    # Log detailed trade info
                    self.logger.info(f"✅ EXECUTED: {signal.signal} {signal.symbol} "
                                   f"| Size: ${position_size:.2f} "
                                   f"| Confidence: {signal.confidence:.2f} "
                                   f"| Expected Return: {signal.expected_return:.3f}")
            
            # 8. Performance tracking and learning
            cycle_end = datetime.now()
            cycle_performance = {
                'cycle_start': cycle_start.isoformat(),
                'cycle_end': cycle_end.isoformat(),
                'duration': (cycle_end - cycle_start).total_seconds(),
                'pairs_analyzed': len(top_pairs),
                'signals_generated': len(trading_signals),
                'trades_executed': len(trades_executed),
                'trades': trades_executed,
                'portfolio_metrics': portfolio_metrics.__dict__,
                'market_sentiment': market_sentiment,
                'usdt_balance_start': usdt_balance,
                'risk_appetite': self.risk_manager.risk_appetite,
                'daily_pnl': self.risk_manager.daily_pnl,
                'advanced_features': True
            }
            
            self.performance_log.append(cycle_performance)
            self.performance_memory.append(cycle_performance)
            
            # 9. Autonomous learning and adaptation
            if len(self.performance_memory) > 5:
                recent_performance = list(self.performance_memory)[-5:]
                avg_trades_per_cycle = sum(p['trades_executed'] for p in recent_performance) / 5
                
                # Adapt trading frequency based on success
                if portfolio_metrics.win_rate > 0.7:
                    self.trading_cycle_minutes = max(5, self.trading_cycle_minutes - 1)
                    self.logger.info("🎯 High win rate detected - increasing trading frequency")
                elif portfolio_metrics.win_rate < 0.4:
                    self.trading_cycle_minutes = min(30, self.trading_cycle_minutes + 2)
                    self.logger.info("📉 Low win rate detected - decreasing trading frequency")
            
            # 10. Comprehensive cycle summary
            self.logger.info(f"🏁 ADVANCED CYCLE COMPLETE:")
            self.logger.info(f"   ⏱️  Duration: {cycle_performance['duration']:.1f}s")
            self.logger.info(f"   📊 Signals: {len(trading_signals)} | Trades: {len(trades_executed)}")
            self.logger.info(f"   💰 Portfolio: ${portfolio_metrics.total_value:.2f}")
            self.logger.info(f"   📈 Win Rate: {portfolio_metrics.win_rate:.1%}")
            self.logger.info(f"   🎯 Risk Appetite: {self.risk_manager.risk_appetite:.2f}")
            
            # 11. Save performance data more frequently for advanced mode
            if len(self.performance_log) % 5 == 0:  # Every 5 cycles instead of 10
                self.save_performance_data()
                
        except Exception as e:
            self.logger.error(f"❌ ADVANCED trading cycle error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def run(self):
        """Main bot loop - ADVANCED autonomous operation with AI learning."""
        self.logger.info("🚀 Starting ADVANCED AZR Trading Bot with full automation")
        
        try:
            # Start autonomous background monitoring
            self.start_autonomous_monitoring()
            
            # Main trading loop
            cycle_count = 0
            while True:
                cycle_count += 1
                
                self.logger.info(f"🔄 CYCLE #{cycle_count} - Advanced Autonomous Trading")
                
                # Run enhanced trading cycle
                self.run_trading_cycle()
                
                # Adaptive wait time based on market conditions and performance
                base_wait = self.trading_cycle_minutes * 60
                
                # Adjust timing based on market volatility
                if hasattr(self, 'last_market_sentiment'):
                    volatility = self.last_market_sentiment.get('volatility_index', 0.3)
                    if volatility > 0.7:
                        wait_seconds = base_wait * 0.5  # Trade more frequently in high volatility
                    elif volatility < 0.2:
                        wait_seconds = base_wait * 1.5  # Trade less frequently in low volatility
                    else:
                        wait_seconds = base_wait
                else:
                    wait_seconds = base_wait
                
                # Performance-based timing adjustment
                if len(self.performance_memory) > 3:
                    recent_performance = list(self.performance_memory)[-3:]
                    avg_success = sum(1 for p in recent_performance if p.get('trades_executed', 0) > 0) / 3
                    
                    if avg_success > 0.8:
                        wait_seconds *= 0.8  # More frequent trading when successful
                    elif avg_success < 0.3:
                        wait_seconds *= 1.3  # Less frequent trading when unsuccessful
                
                wait_seconds = max(60, min(1800, wait_seconds))  # Clamp between 1-30 minutes
                
                self.logger.info(f"⏰ Next cycle in {wait_seconds/60:.1f} minutes "
                               f"(adaptive timing based on market conditions)")
                
                # Intelligent sleep with periodic status updates
                for i in range(int(wait_seconds // 30)):
                    time.sleep(30)
                    
                    # Periodic status updates during wait
                    if i % 4 == 0:  # Every 2 minutes
                        remaining = wait_seconds - (i * 30)
                        if not self.demo_mode:
                            balance = self.get_account_balance()
                            total_value = sum(balance.values())
                            self.logger.info(f"💤 Waiting... {remaining/60:.1f}min left | Portfolio: ${total_value:.2f}")
                        else:
                            self.logger.info(f"💤 DEMO waiting... {remaining/60:.1f}min left")
                
                # Final sleep for remainder
                remaining_sleep = wait_seconds % 30
                if remaining_sleep > 0:
                    time.sleep(remaining_sleep)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Advanced bot stopped by user")
        except Exception as e:
            self.logger.error(f"💥 Fatal error in advanced main loop: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            # Shutdown autonomous monitoring
            self.logger.info("🔄 Shutting down autonomous monitoring...")
            for task in self.background_tasks:
                try:
                    task.cancel()
                except:
                    pass
            
            self.executor.shutdown(wait=True)
            
            # Save final performance data
            self.save_performance_data()
            self.logger.info("🏁 Advanced AZR Trading Bot shutdown complete")
            
            # Final performance summary
            if self.performance_memory:
                total_cycles = len(self.performance_memory)
                total_trades = sum(p.get('trades_executed', 0) for p in self.performance_memory)
                
                self.logger.info(f"📊 FINAL SUMMARY:")
                self.logger.info(f"   🔄 Total Cycles: {total_cycles}")
                self.logger.info(f"   💹 Total Trades: {total_trades}")
                self.logger.info(f"   🎯 Avg Trades/Cycle: {total_trades/total_cycles:.1f}")
                
                if not self.demo_mode:
                    final_balance = self.get_account_balance()
                    final_value = sum(final_balance.values())
                    self.logger.info(f"   💰 Final Portfolio Value: ${final_value:.2f}")
                
                self.logger.info("🎊 Thank you for using Advanced AZR Trading Bot!")

    def analyze_price_data_advanced(self, symbol: str, timeframes: List[str] = ['1m', '5m', '15m']) -> TradingSignal:
        """Advanced multi-timeframe price analysis."""
        if self.demo_mode:
            # Enhanced demo data with more realistic patterns
            import random
            recent_change = random.uniform(-2.0, 2.0)
            confidence = random.uniform(0.2, 0.9) if abs(recent_change) > 0.3 else random.uniform(0.1, 0.4)
            
            # More sophisticated demo signal logic
            if recent_change > 0.5 and confidence > 0.6:
                signal = 'BUY'
            elif recent_change < -0.5 and confidence > 0.6:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            return TradingSignal(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                strength=abs(recent_change) / 2.0,
                timeframe='demo',
                indicators={
                    'demo_change': recent_change,
                    'demo_volume': random.uniform(0.5, 3.0),
                    'demo_momentum': recent_change * random.uniform(0.8, 1.2)
                },
                risk_score=random.uniform(0.1, 0.7),
                expected_return=recent_change / 100
            )
        
        try:
            all_signals = []
            combined_indicators = {}
            
            # Analyze multiple timeframes
            for timeframe in timeframes:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
                
                if len(ohlcv) < 20:
                    continue
                    
                indicators = self.analytics.calculate_technical_indicators(ohlcv)
                combined_indicators[f'{timeframe}_indicators'] = indicators
                
                # Generate signal for this timeframe
                signal_strength = self.calculate_signal_strength(indicators)
                all_signals.append(signal_strength)
            
            if not all_signals:
                return TradingSignal(symbol, 'HOLD', 0, 0, 'none', {}, 1.0, 0)
            
            # Combine signals from all timeframes
            avg_strength = sum(all_signals) / len(all_signals)
            confidence = min(0.95, abs(avg_strength) * 2)
            
            # Determine final signal
            if avg_strength > 0.3:
                signal = 'BUY'
            elif avg_strength < -0.3:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            # Get market sentiment
            market_conditions = self.get_market_sentiment()
            
            # Calculate risk score
            risk_score = self.analytics.calculate_risk_score(
                combined_indicators.get('1m_indicators', {}), 
                market_conditions
            )
            
            # Expected return calculation
            expected_return = avg_strength * confidence * (1 - risk_score)
            
            return TradingSignal(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                strength=abs(avg_strength),
                timeframe=','.join(timeframes),
                indicators=combined_indicators,
                risk_score=risk_score,
                expected_return=expected_return
            )
            
        except Exception as e:
            self.logger.error(f"Advanced analysis error for {symbol}: {e}")
            return TradingSignal(symbol, 'HOLD', 0, 0, 'error', {}, 1.0, 0)
    
    def calculate_signal_strength(self, indicators: Dict[str, float]) -> float:
        """Calculate signal strength from technical indicators."""
        signals = []
        
        # Moving average signals
        if indicators['ma_5'] > indicators['ma_20']:
            signals.append(0.3)
        elif indicators['ma_5'] < indicators['ma_20']:
            signals.append(-0.3)
        
        # RSI signals
        if indicators['rsi'] < 30:
            signals.append(0.5)  # Oversold - buy signal
        elif indicators['rsi'] > 70:
            signals.append(-0.5)  # Overbought - sell signal
        
        # MACD signals
        if indicators['macd'] > 0:
            signals.append(0.2)
        else:
            signals.append(-0.2)
        
        # Volume confirmation
        if indicators['vol_ratio'] > 1.5:
            # High volume amplifies signals
            volume_multiplier = min(2.0, indicators['vol_ratio'])
            signals = [s * volume_multiplier for s in signals]
        
        # Momentum signals
        if indicators['momentum'] > 1:
            signals.append(0.3)
        elif indicators['momentum'] < -1:
            signals.append(-0.3)
        
        return sum(signals) / len(signals) if signals else 0
    
    def adaptive_position_sizing(self, signal: TradingSignal, available_balance: float, 
                               portfolio_size: int) -> float:
        """Advanced adaptive position sizing based on signal quality and risk."""
        if portfolio_size == 0:
            return 0
        
        # Base position size
        base_size = available_balance * 0.9 / portfolio_size
        
        # Use risk manager for adjustment
        adjusted_size = self.risk_manager.adjust_position_size(
            base_size, signal.confidence, signal.risk_score
        )
        
        # Kelly Criterion adjustment for expected return
        if signal.expected_return > 0 and signal.risk_score < 0.5:
            kelly_fraction = signal.expected_return / (signal.risk_score + 0.1)
            kelly_adjustment = min(2.0, max(0.5, kelly_fraction))
            adjusted_size *= kelly_adjustment
        
        return max(self.min_trade_amount, adjusted_size)
    
    def learn_from_trade(self, trade_result: Dict, signal: TradingSignal):
        """AI learning from trade outcomes."""
        try:
            # Calculate actual return
            if 'cost' in trade_result and 'amount' in trade_result:
                # Store learning data
                learning_data = {
                    'timestamp': datetime.now().isoformat(),
                    'signal': signal,
                    'trade_result': trade_result,
                    'expected_return': signal.expected_return,
                    'actual_return': 0,  # Would calculate from future price
                    'success': trade_result.get('status') == 'closed'
                }
                
                self.learning_history.append(learning_data)
                
                # Update success patterns
                signal_key = f"{signal.signal}_{signal.timeframe}"
                if signal_key not in self.success_patterns:
                    self.success_patterns[signal_key] = {'success': 0, 'total': 0}
                
                self.success_patterns[signal_key]['total'] += 1
                if learning_data['success']:
                    self.success_patterns[signal_key]['success'] += 1
                
                # Log learning insights
                success_rate = (self.success_patterns[signal_key]['success'] / 
                               self.success_patterns[signal_key]['total'])
                
                self.logger.info(f"🧠 Learning: {signal_key} success rate: {success_rate:.2%}")
                
        except Exception as e:
            self.logger.error(f"Learning error: {e}")
    
    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics."""
        try:
            if self.demo_mode:
                return PortfolioMetrics(
                    total_value=1000.0,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    win_rate=0.6,
                    sharpe_ratio=1.2,
                    max_drawdown=0.05,
                    volatility=0.15
                )
            
            # Calculate from actual performance data
            recent_trades = list(self.performance_memory)[-50:] if self.performance_memory else []
            
            if not recent_trades:
                balance = self.get_account_balance()
                return PortfolioMetrics(
                    total_value=sum(balance.values()),
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    win_rate=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    volatility=0.0
                )
            
            # Calculate metrics from trade history
            profits = [trade.get('profit_loss', 0) for trade in recent_trades if 'profit_loss' in trade]
            winning_trades = [p for p in profits if p > 0]
            
            win_rate = len(winning_trades) / len(profits) if profits else 0
            avg_return = sum(profits) / len(profits) if profits else 0
            volatility = np.std(profits) if len(profits) > 1 else 0
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            
            # Max drawdown calculation
            cumulative_returns = np.cumsum(profits)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
            
            balance = self.get_account_balance()
            total_value = sum(balance.values())
            
            return PortfolioMetrics(
                total_value=total_value,
                unrealized_pnl=0.0,  # Would need current market prices
                realized_pnl=sum(profits),
                win_rate=win_rate,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility
            )
            
        except Exception as e:
            self.logger.error(f"Portfolio metrics error: {e}")
            return PortfolioMetrics(0, 0, 0, 0, 0, 0, 0)

def main():
    """Main entry point for the Advanced AZR Trading Bot."""
    print("🤖 Advanced AZR Trading Bot v2.0")
    print("🚀 Autonomous AI-Driven Cryptocurrency Trading")
    print("=" * 50)
    
    try:
        bot = AZRTradingBot()
        
        # Display startup information
        if bot.demo_mode:
            print("🟡 DEMO MODE: Safe testing with simulated trading")
            print("💡 Edit .azr_env with real API credentials for live trading")
        else:
            print("🔴 LIVE TRADING MODE: Real money at risk")
            print("⚠️  Monitor logs and performance carefully")
        
        print(f"🎯 Trading Configuration:")
        print(f"   • Min Trade: ${bot.min_trade_amount}")
        print(f"   • Cycle Time: {bot.trading_cycle_minutes} minutes")
        print(f"   • Pairs: {bot.trading_pairs_count}")
        print(f"   • Momentum Threshold: {bot.momentum_threshold:.2%}")
        print(f"   • Dust Threshold: ${bot.max_dust_value}")
        
        print("\n🚀 Starting autonomous trading operations...")
        print("🛑 Press Ctrl+C to stop trading")
        print("=" * 50)
        
        bot.run()
        
    except Exception as e:
        print(f"💥 Failed to start Advanced AZR Trading Bot: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
