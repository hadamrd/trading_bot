#!/usr/bin/env python3
"""
Trader Module - Virtual trading account using existing models
"""

from datetime import datetime
from typing import List, Optional
import sys
from pathlib import Path

# Add src to path to use existing models
sys.path.append(str(Path(__file__).parent.parent / "src"))

from trading_bot.core.models import Position, TradeStatus
from trading_bot.core.enums import OrderSide


class VirtualTrader:
    """Virtual trading account - reuses existing Position model"""
    
    def __init__(self, initial_balance: float = 10000, fee_rate: float = 0.001):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.fee_rate = fee_rate
        
        # Trading state (reuse existing models)
        self.current_position: Optional[Position] = None
        self.closed_positions: List[Position] = []
        
        # Current market prices
        self.current_prices: dict[str, float] = {}
        
        # Statistics
        self.trade_count = 0
    
    def update_price(self, symbol: str, price: float):
        """Update current market price and check position exits"""
        self.current_prices[symbol] = price
        
        # Update position and check for exits
        if self.current_position and self.current_position.symbol == symbol:
            # Update position with new price (existing method)
            row = {"close": price}  # Simulate market data row
            self.current_position.update(row)
            
            # Check if we should close the position
            self._check_position_exit(price)
    
    def buy(self, symbol: str, price: float, amount: float, 
            stop_loss_pct: float = 0.02, reason: str = "") -> bool:
        """Open a buy position using existing Position model"""
        
        # Check if we already have a position
        if self.current_position and self.current_position.is_open:
            print(f"❌ Already have open position in {self.current_position.symbol}")
            return False
        
        # Check balance
        total_cost = amount
        if total_cost > self.balance:
            print(f"❌ Insufficient balance: need ${total_cost:.2f}, have ${self.balance:.2f}")
            return False
        
        # Calculate amount bought after fees (existing logic)
        amount_bought = amount / (price * (1 + self.fee_rate))
        
        # Create position using existing model
        self.current_position = Position(
            symbol=symbol,
            open_time=datetime.now(),
            open_price=price,
            amount_invested=amount,
            amount_bought=amount_bought,
            highest_since_purchase=price,
            buy_reason=reason,
            fee_rate=self.fee_rate,
            stop_loss=stop_loss_pct,
            status=TradeStatus.OPEN
        )
        
        # Update balance
        self.balance -= amount
        self.trade_count += 1
        
        print(f"✅ BUY: {amount_bought:.4f} {symbol} @ ${price:.2f} (${amount:.2f})")
        return True
    
    def sell(self, reason: str = "Manual") -> bool:
        """Close current position"""
        
        if not self.current_position or not self.current_position.is_open:
            print("❌ No open position to sell")
            return False
        
        symbol = self.current_position.symbol
        current_price = self.current_prices.get(symbol)
        
        if not current_price:
            print(f"❌ No current price for {symbol}")
            return False
        
        # Close position using existing method
        liquidation_value = self.current_position.close_position(
            close_time=datetime.now(),
            close_price=current_price,
            reason=reason
        )
        
        # Update balance
        self.balance += liquidation_value
        
        # Move to closed positions
        self.closed_positions.append(self.current_position)
        
        print(f"✅ SELL: {self.current_position.symbol} @ ${current_price:.2f} "
              f"| P&L: ${self.current_position.profit:+.2f} | {reason}")
        
        # Clear current position
        self.current_position = None
        return True
    
    def _check_position_exit(self, current_price: float):
        """Check if current position should be closed"""
        
        if not self.current_position:
            return
        
        # Check stop loss (using existing property)
        entry_price = self.current_position.open_price
        stop_loss_price = entry_price * (1 - self.current_position.stop_loss)
        
        if current_price <= stop_loss_price:
            self.sell("Stop Loss")
            return
        
        # Simple take profit (2x stop loss)
        take_profit_price = entry_price * (1 + (2 * self.current_position.stop_loss))
        if current_price >= take_profit_price:
            self.sell("Take Profit")
            return
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        portfolio_value = self.balance
        
        if self.current_position:
            portfolio_value += self.current_position.liquidation_value
        
        return portfolio_value
    
    def get_stats(self) -> dict:
        """Get trading statistics using existing Position methods"""
        
        if not self.closed_positions:
            return {
                'balance': self.balance,
                'portfolio_value': self.get_portfolio_value(),
                'total_return_pct': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'open_position': self.current_position.symbol if self.current_position else None
            }
        
        # Calculate stats using existing Position properties
        total_profit = sum(pos.profit for pos in self.closed_positions)
        winning_trades = sum(1 for pos in self.closed_positions if pos.profit > 0)
        win_rate = winning_trades / len(self.closed_positions)
        
        portfolio_value = self.get_portfolio_value()
        total_return_pct = ((portfolio_value / self.initial_balance) - 1) * 100
        
        return {
            'balance': self.balance,
            'portfolio_value': portfolio_value,
            'total_return_pct': total_return_pct,
            'total_profit': total_profit,
            'total_trades': len(self.closed_positions),
            'win_rate': win_rate,
            'open_position': self.current_position.symbol if self.current_position else None,
            'avg_trade_pnl': total_profit / len(self.closed_positions) if self.closed_positions else 0
        }
    
    def can_buy(self, amount: float) -> bool:
        """Check if we can make a buy order"""
        return (not self.current_position or not self.current_position.is_open) and amount <= self.balance


# Example usage
def test_trader():
    """Test the virtual trader"""
    
    trader = VirtualTrader(initial_balance=1000)
    
    # Simulate some trades
    trader.update_price("BTCUSDT", 50000)
    trader.buy("BTCUSDT", 50000, 100, reason="Test buy")  # $100 position
    
    # Price moves up
    trader.update_price("BTCUSDT", 51000)  # 2% gain, should trigger take profit
    
    # Show stats
    stats = trader.get_stats()
    print(f"\n📊 Trading Stats:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")


if __name__ == "__main__":
    test_trader()