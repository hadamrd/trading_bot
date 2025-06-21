"""
Backtesting engine for trading strategies.
Migrated from the original backtester.py with clean structure.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from ..core.models import BacktestConfig, BacktestResult, Position, TradingConfig
from ..core.enums import TradeStatus, OrderSide
from ..strategies.base import BaseStrategy
from ..data.market_data import MarketDataManager


class BacktestEngine:
    """
    Main backtesting engine that runs strategies against historical data.
    """
    
    def __init__(self, config: BacktestConfig, strategy: BaseStrategy):
        self.config = config
        self.strategy = strategy
        self.data_manager = MarketDataManager()
        
        # Trading state
        self.balance = config.initial_balance
        self.positions: List[Position] = []
        self.current_position: Optional[Position] = None
        self.trade_count = 0
        
    def run(self) -> Dict[str, BacktestResult]:
        """Run backtest for all symbols in config."""
        results = {}
        
        for symbol in self.config.symbols:
            print(f"📊 Backtesting {symbol}...")
            result = self.backtest_symbol(symbol)
            results[symbol] = result
            
            # Reset state for next symbol
            self.reset_state()
        
        return results
    
    def backtest_symbol(self, symbol: str) -> BacktestResult:
        """Run backtest for a single symbol."""
        start_time = datetime.now()
        
        # Get market data
        df = self.data_manager.get_data_for_backtest(
            symbol=symbol,
            timeframe=self.config.timeframe,
            start_date=self.config.since_date,
            end_date=self.config.test_end_date,
            with_indicators=True
        )
        
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Prepare data with strategy indicators (use all data for proper indicator calculation)
        df = self.strategy.prepare_data(df)
        
        # Filter to test period only (but keep all data for indicator calculation)
        test_df = df[
            (df.index >= self.config.test_start_date) & 
            (df.index <= self.config.test_end_date)
        ]
        
        if test_df.empty:
            print(f"⚠️  No data in test period for {symbol}")
            return self.generate_backtest_result(symbol, start_time, end_time, 0)
        
        print(f"📈 Processing {len(test_df)} candles for {symbol} (test period)")
        print(f"   Using {len(df)} total candles for indicators")
        
        # Run the backtest - EXACT same logic as original backtester.py
        previous_row = None
        for _, row in test_df.iterrows():
            if previous_row is not None:
                self.process_timestamp(row, previous_row, symbol)
            previous_row = row
        
        # Close any remaining position
        if self.current_position and self.current_position.is_open:
            last_row = test_df.iloc[-1]
            self.close_position(
                close_time=last_row.name,
                close_price=Decimal(str(last_row['close'])),
                reason="End of backtest"
            )
        
        # Generate results
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        return self.generate_backtest_result(symbol, start_time, end_time, execution_time)
    
    def process_dataframe(self, df: pd.DataFrame, symbol: str):
        """Process each candle in the dataframe."""
        previous_row = None
        
        for timestamp, row in df.iterrows():
            # Add symbol and timestamp to row for strategy
            row_dict = row.to_dict()
            row_dict['symbol'] = symbol
            row_dict['timestamp'] = timestamp
            row_series = pd.Series(row_dict)
            
            # Update current position with new price data
            if self.current_position and self.current_position.is_open:
                self.current_position.update(Decimal(str(row['close'])))
                
                # Check if we should close the position
                close_reason = self.strategy.should_close_position(self.current_position, row_series)
                if close_reason:
                    self.close_position(
                        close_time=timestamp,
                        close_price=Decimal(str(row['close'])),
                        reason=close_reason
                    )
            
            # Check for new signals (only if no open position)
            if not self.current_position or not self.current_position.is_open:
                signal = self.strategy.generate_signal(
                    current_bar=row_series,
                    previous_bar=self.series_from_row(previous_row, symbol) if previous_row is not None else None,
                    position=None
                )
                
                if signal and signal.action == OrderSide.BUY:
                    self.open_position(
                        symbol=symbol,
                        timestamp=timestamp,
                        price=Decimal(str(row['close'])),
                        signal=signal
                    )
            
            previous_row = row
    
    def open_position(self, symbol: str, timestamp: datetime, price: Decimal, signal):
        """Open a new trading position."""
        if self.current_position and self.current_position.is_open:
            return  # Can't open if we already have an open position
        
        # Calculate position size
        position_size = self.strategy.calculate_position_size(
            available_balance=self.balance,
            current_price=price,
            market_data=pd.Series()  # TODO: Pass actual market data
        )
        
        # Create position
        amount_bought = position_size / price
        
        self.current_position = Position(
            symbol=symbol,
            side=signal.action,
            open_time=timestamp,
            open_price=price,
            amount_invested=position_size,
            amount_bought=amount_bought,
            highest_since_purchase=price,
            buy_reason=signal.reason,
            fee_rate=self.config.fee_rate,
            stop_loss=Decimal("0")  # Will be set below
        )
        
        # Calculate stop loss and take profit
        stop_loss_price = self.strategy.calculate_stop_loss(
            entry_price=price,
            side=signal.action,
            market_data=pd.Series()
        )
        
        take_profit_price = self.strategy.calculate_take_profit(
            entry_price=price,
            side=signal.action,
            market_data=pd.Series()
        )
        
        # Set stop loss
        if stop_loss_price:
            self.current_position.stop_loss = stop_loss_price
        
        # Store take profit for reference (Position model doesn't have this field)
        self.current_position.take_profit = take_profit_price
        
        # Update balance (subtract fees)
        fees = position_size * self.config.fee_rate
        self.balance -= fees
        
        self.trade_count += 1
        print(f"🟢 OPEN {signal.action} {symbol} @ ${price:.4f} | Size: ${position_size:.2f} | Reason: {signal.reason}")
    
    def close_position(self, close_time: datetime, close_price: Decimal, reason: str):
        """Close the current position."""
        if not self.current_position or not self.current_position.is_open:
            return
        
        # Close the position
        liquidation_value = self.current_position.close_position(close_time, close_price, reason)
        
        # Update balance
        fees = liquidation_value * self.config.fee_rate
        self.balance += liquidation_value - fees
        
        # Add to completed positions
        self.positions.append(self.current_position)
        
        profit = self.current_position.net_pnl
        profit_pct = self.current_position.return_percentage
        
        print(f"🔴 CLOSE {self.current_position.symbol} @ ${close_price:.4f} | "
              f"P&L: ${profit:.2f} ({profit_pct:.2f}%) | Reason: {reason}")
        
        self.current_position = None
    
    def generate_backtest_result(self, symbol: str, start_time: datetime, end_time: datetime, execution_time: float) -> BacktestResult:
        """Generate backtest results."""
        if not self.positions:
            return BacktestResult(
                symbol=symbol,
                strategy_name=self.strategy.name,
                config=self.config,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_profit=Decimal("0"),
                win_rate=0.0,
                initial_balance=self.config.initial_balance,
                final_balance=self.balance,
                total_return=Decimal("0"),
                total_return_pct=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                profit_factor=0.0,
                average_profit=Decimal("0"),
                average_loss=Decimal("0"),
                largest_win=Decimal("0"),
                largest_loss=Decimal("0"),
                average_holding_time=0.0,
                trades=self.positions,
                start_time=start_time,
                end_time=end_time,
                execution_time_seconds=execution_time
            )
        
        # Calculate metrics
        total_trades = len(self.positions)
        winning_trades = len([p for p in self.positions if p.net_pnl > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        wins = [p.net_pnl for p in self.positions if p.net_pnl > 0]
        losses = [p.net_pnl for p in self.positions if p.net_pnl < 0]
        
        avg_win = Decimal(str(np.mean(wins))) if wins else Decimal("0")
        avg_loss = Decimal(str(np.mean(losses))) if losses else Decimal("0")
        largest_win = max(wins) if wins else Decimal("0")
        largest_loss = min(losses) if losses else Decimal("0")
        
        total_return = self.balance - self.config.initial_balance
        total_return_pct = float((total_return / self.config.initial_balance) * 100)
        
        # Calculate average holding time
        holding_times = [p.duration_hours for p in self.positions if p.duration_hours is not None]
        avg_holding_time = float(np.mean(holding_times)) if holding_times else 0.0
        
        # Calculate profit factor
        gross_profit = sum(wins) if wins else Decimal("0")
        gross_loss = abs(sum(losses)) if losses else Decimal("0")
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        # Simple drawdown calculation
        returns = [float(p.return_percentage) for p in self.positions]
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        
        # Simple Sharpe ratio calculation
        if len(returns) > 1:
            sharpe_ratio = float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        return BacktestResult(
            symbol=symbol,
            strategy_name=self.strategy.name,
            config=self.config,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_profit=total_return,  # Same as total_return for now
            win_rate=win_rate,
            initial_balance=self.config.initial_balance,
            final_balance=self.balance,
            total_return=total_return,
            total_return_pct=total_return_pct,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sharpe_ratio,  # Simplified
            profit_factor=profit_factor,
            average_profit=avg_win,
            average_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            average_holding_time=avg_holding_time,
            trades=self.positions,
            start_time=start_time,
            end_time=end_time,
            execution_time_seconds=execution_time
        )
    
    def reset_state(self):
        """Reset trading state for next symbol."""
        self.balance = self.config.initial_balance
        self.positions = []
        self.current_position = None
        self.trade_count = 0
    
    def series_from_row(self, row: Optional[pd.Series], symbol: str) -> Optional[pd.Series]:
        """Convert row to series with symbol."""
        if row is None:
            return None
        
        row_dict = row.to_dict()
        row_dict['symbol'] = symbol
        return pd.Series(row_dict)
    
    def print_summary(self, results: Dict[str, BacktestResult]):
        """Print a summary of backtest results."""
        print(f"\n📊 BACKTEST SUMMARY")
        print(f"=" * 50)
        
        for symbol, result in results.items():
            print(f"\n{symbol}:")
            print(f"  Total Trades: {result.total_trades}")
            print(f"  Win Rate: {result.win_rate:.1%}")
            print(f"  Total Return: ${result.total_return:.2f} ({result.total_return_pct:.2f}%)")
            print(f"  Max Drawdown: {result.max_drawdown:.2f}%")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"  Profit Factor: {result.profit_factor:.2f}")