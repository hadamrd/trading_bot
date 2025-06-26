#!/usr/bin/env python3
"""
Trading Engine - Orchestrates broker, strategy, and trader
"""

import asyncio
import signal
from datetime import datetime

from trading_bot.order_book_trading.producer import BinanceBroker
from trading_bot.order_book_trading.virtual_trader import VirtualTrader
from trading_bot.order_book_trading.strats.base import BaseStrategy
from trading_bot.order_book_trading.models import MarketSituation


class LiveTradingEngine:
    """Main trading engine that coordinates all components"""
    
    def __init__(self, 
                 strategy: BaseStrategy,
                 initial_balance: float = 1000,
                 stats_interval: int = 30):  # Show stats every 30 seconds
        
        self.strategy = strategy
        self.stats_interval = stats_interval
        
        # Initialize components
        self.broker = BinanceBroker()
        self.trader = VirtualTrader(initial_balance=initial_balance)
        
        # Wire strategy to trader
        self.strategy.set_trader(self.trader)
        
        # Add strategy as broker callback
        self.broker.add_callback(self._on_market_update)
        
        # Stats tracking
        self.last_stats_time = datetime.now()
        self.update_count = 0
        self.running = False
    
    def _on_market_update(self, situation: MarketSituation):
        """Handle market updates from broker"""
        
        self.update_count += 1
        
        # Pass to strategy
        self.strategy.on_market_update(situation)
        
        # Show periodic stats
        self._maybe_show_stats()
    
    def _maybe_show_stats(self):
        """Show stats periodically"""
        
        now = datetime.now()
        seconds_since_last = (now - self.last_stats_time).total_seconds()
        
        if seconds_since_last >= self.stats_interval:
            self._show_stats()
            self.last_stats_time = now
    
    def _show_stats(self):
        """Display current trading statistics"""
        
        stats = self.strategy.get_stats()
        
        print(f"\n📊 Trading Stats ({datetime.now().strftime('%H:%M:%S')})")
        print(f"   Updates received: {self.update_count:,}")
        print(f"   Portfolio value: ${stats.get('portfolio_value', 0):.2f}")
        print(f"   Total return: {stats.get('total_return_pct', 0):+.2f}%")
        print(f"   Open position: {stats.get('open_position', 'None')}")
        print(f"   Total trades: {stats.get('total_trades', 0)}")
        
        if stats.get('total_trades', 0) > 0:
            print(f"   Win rate: {stats.get('win_rate', 0):.1%}")
            print(f"   Avg trade P&L: ${stats.get('avg_trade_pnl', 0):+.2f}")
    
    async def start(self, symbol: str = "BTCUSDT"):
        """Start the trading engine"""
        
        print(f"🚀 Starting Live Trading Engine")
        print(f"   Strategy: {self.strategy.name}")
        print(f"   Symbol: {symbol}")
        print(f"   Initial balance: ${self.trader.initial_balance:,.2f}")
        print(f"   Stats interval: {self.stats_interval}s")
        print("-" * 50)
        
        self.running = True
        
        # Setup graceful shutdown
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}, stopping...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Start broker stream
            await self.broker.start_stream(symbol)
        except Exception as e:
            print(f"❌ Engine error: {e}")
        finally:
            self._show_final_stats()
    
    def stop(self):
        """Stop the trading engine"""
        self.running = False
        self.broker.stop()
    
    def _show_final_stats(self):
        """Show final statistics"""
        
        print(f"\n🏁 Final Trading Results")
        print("=" * 50)
        
        stats = self.strategy.get_stats()
        
        print(f"Strategy: {stats.get('strategy_name', 'Unknown')}")
        print(f"Initial balance: ${self.trader.initial_balance:,.2f}")
        print(f"Final portfolio: ${stats.get('portfolio_value', 0):,.2f}")
        print(f"Total return: {stats.get('total_return_pct', 0):+.2f}%")
        print(f"Total profit: ${stats.get('total_profit', 0):+.2f}")
        print(f"Total trades: {stats.get('total_trades', 0)}")
        
        if stats.get('total_trades', 0) > 0:
            print(f"Win rate: {stats.get('win_rate', 0):.1%}")
            print(f"Avg trade P&L: ${stats.get('avg_trade_pnl', 0):+.2f}")
        
        print(f"Market updates: {self.update_count:,}")


# Test function
async def run_test_engine():
    """Run a test of the trading engine"""
    
    # Import strategy here to avoid circular imports
    from trading_bot.order_book_trading.strats.simple_strat import SimpleOrderBookStrategy
    
    # Create strategy
    strategy = SimpleOrderBookStrategy(
        position_size=50,  # $50 per trade for testing
        min_wall_size=25000,  # Lower threshold for more activity
        min_pressure_diff=0.3  # More sensitive to pressure changes
    )
    
    # Create and start engine
    engine = LiveTradingEngine(
        strategy=strategy,
        initial_balance=500,  # Start with $500
        stats_interval=20  # Stats every 20 seconds
    )
    
    try:
        await engine.start("WIFUSDT")
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user")
        engine.stop()


if __name__ == "__main__":
    print("🧪 Running Trading Engine Test")
    print("Press Ctrl+C to stop")
    asyncio.run(run_test_engine())
