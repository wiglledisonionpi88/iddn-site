#!/usr/bin/env python3
"""
Paper Trading Mode for AZR Trading Bot
Test the bot without real money
"""

import json
import time
from datetime import datetime
from azr_trading_bot import AZRTradingBot

class PaperTradingBot(AZRTradingBot):
    def __init__(self):
        super().__init__()
        self.paper_balance = {'USDT': 1000}  # Start with $1000 USDT
        self.paper_trades = []
        self.logger.info("Paper Trading Mode - Starting with $1000 USDT")
    
    def get_account_balance(self):
        """Override to return paper trading balance."""
        return self.paper_balance.copy()
    
    def execute_trade(self, symbol, signal, position_size):
        """Override to simulate trades without real execution."""
        if signal == 'HOLD':
            return None
        
        try:
            # Get current price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Simulate trade execution
            if signal == 'BUY' and self.paper_balance.get('USDT', 0) >= position_size:
                # Buy simulation
                amount = position_size / current_price
                fee = position_size * 0.001  # 0.1% fee
                
                self.paper_balance['USDT'] -= position_size + fee
                base_currency = symbol.split('/')[0]
                self.paper_balance[base_currency] = self.paper_balance.get(base_currency, 0) + amount
                
                trade_log = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'signal': signal,
                    'amount': amount,
                    'cost': position_size,
                    'price': current_price,
                    'fee': fee,
                    'type': 'PAPER_TRADE'
                }
                
                self.paper_trades.append(trade_log)
                self.logger.info(f"PAPER TRADE: {signal} {amount:.6f} {base_currency} at {current_price:.6f} USDT")
                return trade_log
                
            elif signal == 'SELL':
                base_currency = symbol.split('/')[0]
                amount = self.paper_balance.get(base_currency, 0)
                
                if amount > 0:
                    usdt_received = amount * current_price
                    fee = usdt_received * 0.001  # 0.1% fee
                    
                    self.paper_balance[base_currency] = 0
                    self.paper_balance['USDT'] = self.paper_balance.get('USDT', 0) + usdt_received - fee
                    
                    trade_log = {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'signal': signal,
                        'amount': amount,
                        'cost': usdt_received,
                        'price': current_price,
                        'fee': fee,
                        'type': 'PAPER_TRADE'
                    }
                    
                    self.paper_trades.append(trade_log)
                    self.logger.info(f"PAPER TRADE: {signal} {amount:.6f} {base_currency} for {usdt_received:.2f} USDT")
                    return trade_log
        
        except Exception as e:
            self.logger.error(f"Error in paper trade for {symbol}: {e}")
        
        return None
    
    def consolidate_dust(self, dust_coins):
        """Override to simulate dust consolidation."""
        total_value = 0
        for currency, amount in dust_coins:
            try:
                symbol = f"{currency}/USDT"
                ticker = self.exchange.fetch_ticker(symbol)
                value = amount * ticker['last']
                total_value += value
                
                # Simulate selling dust
                self.paper_balance[currency] = 0
                self.paper_balance['USDT'] = self.paper_balance.get('USDT', 0) + value
                
                self.logger.info(f"PAPER DUST: Sold {amount} {currency} for {value:.4f} USDT")
                
            except Exception as e:
                self.logger.error(f"Error simulating dust consolidation for {currency}: {e}")
        
        return total_value
    
    def save_performance_data(self):
        """Override to include paper trading data."""
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'mode': 'PAPER_TRADING',
            'paper_balance': self.paper_balance,
            'paper_trades': self.paper_trades,
            'trading_data': self.trading_data,
            'performance_log': self.performance_log
        }
        
        filename = f"data/paper_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(performance_data, f, indent=2)
            self.logger.info(f"Paper trading data saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving paper trading data: {e}")

def main():
    """Main entry point for paper trading."""
    try:
        bot = PaperTradingBot()
        bot.run()
    except Exception as e:
        print(f"Failed to start Paper Trading Bot: {e}")

if __name__ == "__main__":
    main()
