from trading_bot.order_book_trading.models import MarketSituation
from trading_bot.order_book_trading.strats.base import BaseStrategy


class SimpleOrderBookStrategy(BaseStrategy):
    """Simple strategy based on order book walls and pressure"""
    
    def __init__(self, 
                 position_size: float = 100,  # $100 per trade
                 min_wall_size: float = 100000,  # Minimum wall size to consider
                 min_pressure_diff: float = 0.15,  # Minimum pressure difference
                 cooldown_seconds: int = 60,  # Wait between trades
                 min_hold_seconds: int = 30):  # Minimum position hold time
        
        super().__init__("SimpleOrderBook")
        self.position_size = position_size
        self.min_wall_size = min_wall_size 
        self.min_pressure_diff = min_pressure_diff
        self.cooldown_seconds = cooldown_seconds
        self.min_hold_seconds = min_hold_seconds
        
        # Track last few signals to avoid noise
        self.last_signals = []
        self.max_signal_history = 5  # More confirmation needed
        
        # Timing controls
        self.last_trade_time = None
        self.position_open_time = None
    
    def on_market_update(self, situation: MarketSituation):
        """Analyze market situation and make trading decisions"""
        
        if not self.trader:
            return
        
        # Update trader with current price
        self.trader.update_price(situation.symbol, situation.price)
        
        # Generate trading signal
        signal = self._generate_signal(situation)
        
        # Track signal history
        self.last_signals.append(signal)
        if len(self.last_signals) > self.max_signal_history:
            self.last_signals.pop(0)
        
        # Execute trades based on signal (with timing controls)
        self._execute_signal(situation, signal)
    
    def _generate_signal(self, situation: MarketSituation) -> str:
        """Generate BUY/SELL/HOLD signal with stricter criteria"""
        
        # Check for large walls (much higher threshold)
        has_large_bid_wall = (situation.large_bid_wall and 
                             situation.large_bid_wall.value >= self.min_wall_size)
        has_large_ask_wall = (situation.large_ask_wall and 
                             situation.large_ask_wall.value >= self.min_wall_size)
        
        # Check bid pressure (more extreme thresholds)
        very_strong_bid_pressure = situation.bid_pressure > (0.5 + self.min_pressure_diff)
        very_weak_bid_pressure = situation.bid_pressure < (0.5 - self.min_pressure_diff)
        
        # Conflicting signals = HOLD (avoid whipsaws)
        if (has_large_bid_wall and has_large_ask_wall):
            return "HOLD"
        
        if (very_strong_bid_pressure and very_weak_bid_pressure):
            return "HOLD"
        
        # Generate signal only on clear conditions
        if has_large_bid_wall and very_strong_bid_pressure:
            return "BUY"  # Strong confluence
        elif has_large_ask_wall and very_weak_bid_pressure:
            return "SELL"  # Strong confluence  
        elif has_large_bid_wall or very_strong_bid_pressure:
            return "BUY"  # Single strong signal
        elif has_large_ask_wall or very_weak_bid_pressure:
            return "SELL"  # Single strong signal
        else:
            return "HOLD"
    
    def _execute_signal(self, situation: MarketSituation, signal: str):
        """Execute trading signal with timing controls"""
        
        now = situation.timestamp
        
        # Check cooldown period
        if self.last_trade_time:
            seconds_since_trade = (now - self.last_trade_time).total_seconds()
            if seconds_since_trade < self.cooldown_seconds:
                return  # Still in cooldown
        
        # Need strong signal confirmation (most recent signals must agree)
        if len(self.last_signals) < self.max_signal_history:
            return
        
        recent_signals = self.last_signals[-3:]  # Last 3 signals must agree
        
        # BUY logic
        if (signal == "BUY" and 
            all(s in ["BUY", "HOLD"] for s in recent_signals) and  # No conflicting SELL signals
            recent_signals.count("BUY") >= 2 and  # At least 2 BUY signals
            self.trader.can_buy(self.position_size)):
            
            reason = self._get_buy_reason(situation)
            success = self.trader.buy(
                symbol=situation.symbol,
                price=situation.price,
                amount=self.position_size,
                reason=reason
            )
            
            if success:
                self.last_trade_time = now
                self.position_open_time = now
        
        # SELL logic - only if position held long enough
        elif (signal == "SELL" and 
              all(s in ["SELL", "HOLD"] for s in recent_signals) and  # No conflicting BUY signals
              recent_signals.count("SELL") >= 2 and  # At least 2 SELL signals
              self.trader.current_position and 
              self.trader.current_position.is_open):
            
            # Check minimum hold time
            if self.position_open_time:
                hold_time = (now - self.position_open_time).total_seconds()
                if hold_time < self.min_hold_seconds:
                    return  # Position too new
            
            reason = self._get_sell_reason(situation)
            success = self.trader.sell(reason=reason)
            
            if success:
                self.last_trade_time = now
                self.position_open_time = None
    
    def _get_buy_reason(self, situation: MarketSituation) -> str:
        """Get reason for buy signal"""
        reasons = []
        
        if situation.large_bid_wall:
            reasons.append(f"Large bid wall ${situation.large_bid_wall.value:,.0f}")
        
        if situation.bid_pressure > 0.7:
            reasons.append(f"Strong bid pressure {situation.bid_pressure:.1%}")
        
        return " | ".join(reasons) if reasons else "Order book signal"
    
    def _get_sell_reason(self, situation: MarketSituation) -> str:
        """Get reason for sell signal"""
        reasons = []
        
        if situation.large_ask_wall:
            reasons.append(f"Large ask wall ${situation.large_ask_wall.value:,.0f}")
        
        if situation.bid_pressure < 0.3:
            reasons.append(f"Weak bid pressure {situation.bid_pressure:.1%}")
        
        return " | ".join(reasons) if reasons else "Order book signal"
    
    def get_stats(self) -> dict:
        """Get strategy statistics"""
        base_stats = {
            'strategy_name': self.name,
            'position_size': self.position_size,
            'min_wall_size': self.min_wall_size,
            'min_pressure_diff': self.min_pressure_diff
        }
        
        if self.trader:
            trader_stats = self.trader.get_stats()
            return {**base_stats, **trader_stats}
        
        return base_stats
