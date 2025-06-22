"""
Backtesting engine for trading strategies.
Fixed to match the original working TradingBot interface.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd

from ..core.enums import TradeStatus
from ..core.models import BacktestConfig, BacktestResult, Position
from ..data.market_data import MarketDataManager
from ..strategies.base import BaseStrategy


class BacktestEngine:
    """
    Main backtesting engine that runs strategies against historical data.
    Restored to match original TradingBot interface.
    """

    def __init__(self, config: BacktestConfig, strategy: BaseStrategy):
        self.config = config
        self.strategy = strategy
        self.data_manager = MarketDataManager()

        # Trading state - match original TradingBot
        self.balance = float(config.initial_balance)
        self.initial_balance = float(config.initial_balance)
        self.positions: list[Position] = []  # closed_trades in old code
        self.current_position: Position | None = None  # open_position in old code
        self.trade_count = 0

    def run(self) -> dict[str, BacktestResult]:
        """Run backtest for all symbols in config."""
        print(f"🚀 Starting backtest for {len(self.config.symbols)} symbols...")
        print(f"   Period: {self.config.test_start_date.date()} to {(self.config.test_end_date or datetime.now()).date()}")

        results = {}

        for symbol in self.config.symbols:
            print(f"\n📊 Backtesting {symbol}...")
            result = self.backtest_symbol(symbol)
            results[symbol] = result

            # Reset state for next symbol
            self.reset_state()

        return results

    def validate_data_availability(self) -> bool:
        """Check if required data is available for all symbols."""
        missing_symbols = []

        # Define what period we actually need
        required_start = self.config.since_date
        required_end = self.config.test_end_date or datetime.now()

        for symbol in self.config.symbols:
            existing_range = self.data_manager.storage.get_date_range(symbol, self.config.timeframe)

            if not existing_range:
                missing_symbols.append(f"{symbol} (no data)")
                continue

            existing_start, existing_end = existing_range
            has_enough_start = existing_start <= required_start
            has_enough_end = existing_end >= required_end

            if not (has_enough_start and has_enough_end):
                missing_symbols.append(f"{symbol} (need {required_start.date()} to {required_end.date()}, have {existing_start.date()} to {existing_end.date()})")

        if missing_symbols:
            print(f"⚠️  Missing data for: {', '.join(missing_symbols)}")
            return False
        else:
            print(f"✅ All required data available for {len(self.config.symbols)} symbols")
            return True

    def ensure_data_available(self, symbol: str):
        """Ensure required data is available, download only if missing."""
        # Define what date range we actually need
        required_start = self.config.since_date
        required_end = self.config.test_end_date or datetime.now()

        # Check what data we already have
        existing_range = self.data_manager.storage.get_date_range(symbol, self.config.timeframe)

        if existing_range:
            existing_start, existing_end = existing_range

            # Check if existing data covers our required period
            has_enough_start = existing_start <= required_start
            has_enough_end = existing_end >= required_end

            if has_enough_start and has_enough_end:
                print(f"✅ {symbol} data already covers required period ({existing_start.date()} to {existing_end.date()})")
                return
            else:
                print(f"📥 {symbol} data insufficient for required period...")
                print(f"   Need: {required_start.date()} to {required_end.date()}")
                print(f"   Have: {existing_start.date()} to {existing_end.date()}")
        else:
            print(f"📥 {symbol} data not found, downloading...")

        # Calculate how much data to download to cover the required period
        # Add buffer for indicators (50 days should be enough for most indicators)
        download_start = required_start - timedelta(days=50)
        download_end = max(required_end, datetime.now())
        total_days = (download_end - download_start).days + 10  # Extra buffer

        # Download data using the standard method but with explicit date range
        try:
            # Use the binance client directly for more control
            candles = self.data_manager.binance.fetch_historical_data(
                symbol=symbol,
                timeframe=self.config.timeframe,
                days=total_days,
                end_date=download_end
            )

            if candles:
                count = self.data_manager.storage.store_candles(candles)
                print(f"✅ Downloaded {count} candles for {symbol}")

                # Update indicators for the symbol
                self.data_manager.update_indicators(symbol, self.config.timeframe, download_start)
            else:
                print(f"⚠️  No data returned from Binance for {symbol}")

        except Exception as e:
            print(f"⚠️  Warning: Could not download data for {symbol}: {e}")
            print("   Proceeding with existing data...")

    def backtest_symbol(self, symbol: str) -> BacktestResult:
        """Run backtest for a single symbol."""
        start_time = datetime.now()

        # Ensure data is available - download only if missing
        self.ensure_data_available(symbol)

        # Get market data
        df = self.data_manager.get_data_for_backtest(
            symbol=symbol,
            timeframe=self.config.timeframe,
            start_date=self.config.since_date,
            end_date=self.config.test_end_date,
            with_indicators=False  # We'll calculate them with the strategy
        )

        if df.empty:
            raise ValueError(f"No data available for {symbol}")

        # Prepare data with strategy indicators (use all data for proper indicator calculation)
        df = self.strategy.prepare_data(df)

        # Filter to test period only (but keep all data for indicator calculation)
        if self.config.test_end_date is not None:
            test_df = df[
                (df.index >= self.config.test_start_date) &
                (df.index <= self.config.test_end_date)
            ].copy()
        else:
            test_df = df[df.index >= self.config.test_start_date].copy()

        if test_df.empty:
            print(f"⚠️  No data in test period for {symbol}")
            return self.generate_empty_result(symbol, start_time)

        print(f"📈 Processing {len(test_df)} candles for {symbol} (test period)")
        print(f"   Using {len(df)} total candles for indicators")

        # Add timestamp column for compatibility
        test_df['timestamp'] = test_df.index

        # Run the backtest - EXACT same logic as original TradingBot
        previous_row = None
        for _, row in test_df.iterrows():
            if previous_row is not None:
                self.process_timestamp(row, previous_row)
            previous_row = row

        # Close any remaining position - match original logic
        if self.current_position and self.current_position.is_open:
            last_row = test_df.iloc[-1]
            self.sell_asset(last_row, "End of backtest")

        # Generate results
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        return self.generate_backtest_result(symbol, start_time, end_time, execution_time)

    def process_timestamp(self, row: pd.Series, previous_row: pd.Series):
        """
        Core trading logic - EXACT copy from original TradingBot.
        """
        if self.current_position:
            self.current_position.update(row)
            sell, reason = self.strategy.sell_condition(self.current_position, row)
            if sell:
                self.sell_asset(row, reason)
        else:
            buy, reason, stop_loss = self.strategy.buy_condition(row, previous_row)
            if buy:
                self.buy_asset(row, reason, stop_loss)

    def buy_asset(self, row: pd.Series, reason: str, stop_loss: float):
        """
        Buy logic - EXACT copy from original TradingBot.
        """
        position_size = self.strategy.calculate_position_size(self.balance, row["close"], row)
        amount_invested = min(position_size, self.balance)
        self.balance -= amount_invested
        amount_bought = amount_invested / (row["close"] * (1 + float(self.config.fee_rate)))

        # Convert timestamp to datetime if needed
        if isinstance(row.get('timestamp'), pd.Timestamp):
            open_time = row['timestamp'].to_pydatetime()
        else:
            open_time = row.name  # Use index if timestamp column not available

        self.current_position = Position(
            symbol=row.get('symbol', self.config.symbols[0]),  # Use symbol from config if not in row
            open_time=open_time,
            open_price=float(row["close"]),
            amount_invested=amount_invested,
            amount_bought=amount_bought,
            highest_since_purchase=float(row["close"]),
            buy_reason=reason,
            fee_rate=float(self.config.fee_rate),
            stop_loss=stop_loss,
            status=TradeStatus.OPEN
        )

        self.trade_count += 1
        print(f"🟢 BUY @ ${row['close']:.4f} | Size: ${amount_invested:.2f} | Reason: {reason}")

    def sell_asset(self, row: pd.Series, reason: str):
        """
        Sell logic - EXACT copy from original TradingBot.
        """
        if not self.current_position:
            return

        # Convert timestamp to datetime if needed
        if isinstance(row.get('timestamp'), pd.Timestamp):
            close_time = row['timestamp'].to_pydatetime()
        else:
            close_time = row.name  # Use index if timestamp column not available

        liquidation_value = self.current_position.close_position(close_time, float(row["close"]), reason)
        self.positions.append(self.current_position)
        self.balance += liquidation_value

        profit = self.current_position.profit
        print(f"🔴 SELL @ ${row['close']:.4f} | P&L: ${profit:.2f} ({self.current_position.return_percentage:.2f}%) | Reason: {reason}")

        self.current_position = None

    def generate_backtest_result(self, symbol: str, start_time: datetime, end_time: datetime, execution_time: float) -> BacktestResult:
        """Generate backtest results."""
        if not self.positions:
            return self.generate_empty_result(symbol, start_time)

        # Calculate metrics - match original logic
        total_trades = len(self.positions)
        winning_trades = len([p for p in self.positions if p.profit > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        wins = [p.profit for p in self.positions if p.profit > 0]
        losses = [p.profit for p in self.positions if p.profit < 0]

        avg_win = Decimal(str(np.mean(wins))) if wins else Decimal("0")
        avg_loss = Decimal(str(np.mean(losses))) if losses else Decimal("0")
        largest_win = Decimal(str(max(wins))) if wins else Decimal("0")
        largest_loss = Decimal(str(min(losses))) if losses else Decimal("0")

        total_profit = sum(p.profit for p in self.positions)
        total_return = Decimal(str(total_profit))
        total_return_pct = float((total_return / Decimal(str(self.initial_balance))) * 100)

        # Calculate average holding time
        holding_times = [p.duration_hours for p in self.positions]
        avg_holding_time = float(np.mean(holding_times)) if holding_times else 0.0

        # Calculate profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

        # Simple drawdown calculation
        returns = [p.return_percentage / 100 for p in self.positions]  # Convert to decimal
        if returns:
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            max_drawdown = float(np.min(drawdown)) * 100  # Convert back to percentage
        else:
            max_drawdown = 0.0

        # Simple Sharpe ratio calculation
        if len(returns) > 1:
            sharpe_ratio = float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        return BacktestResult(
            symbol=symbol,
            strategy_name=self.strategy.__class__.__name__,
            config=self.config,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_profit=total_return,
            win_rate=win_rate,
            initial_balance=Decimal(str(self.initial_balance)),
            final_balance=Decimal(str(self.balance)),
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

    def generate_empty_result(self, symbol: str, start_time: datetime) -> BacktestResult:
        """Generate empty result when no trades occurred."""
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        return BacktestResult(
            symbol=symbol,
            strategy_name=self.strategy.__class__.__name__,
            config=self.config,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_profit=Decimal("0"),
            win_rate=0.0,
            initial_balance=Decimal(str(self.initial_balance)),
            final_balance=Decimal(str(self.balance)),
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
            trades=[],
            start_time=start_time,
            end_time=end_time,
            execution_time_seconds=execution_time
        )

    def reset_state(self):
        """Reset trading state for next symbol."""
        self.balance = self.initial_balance
        self.positions = []
        self.current_position = None
        self.trade_count = 0

    def portfolio_value(self) -> float:
        """Calculate current portfolio value - match original TradingBot."""
        if self.current_position:
            return self.balance + self.current_position.liquidation_value
        return self.balance

    def print_summary(self, results: dict[str, BacktestResult]):
        """Print a summary of backtest results."""
        print("\n📊 BACKTEST SUMMARY")
        print("=" * 50)

        for symbol, result in results.items():
            print(f"\n{symbol}:")
            print(f"  Total Trades: {result.total_trades}")
            print(f"  Win Rate: {result.win_rate:.1%}")
            print(f"  Total Return: ${result.total_return:.2f} ({result.total_return_pct:.2f}%)")
            print(f"  Max Drawdown: {result.max_drawdown:.2f}%")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"  Profit Factor: {result.profit_factor:.2f}")
