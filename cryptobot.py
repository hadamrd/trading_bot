import threading
import time
from typing import List, Optional
import pandas as pd
from datetime import datetime
from tradingbot2.lib.BinancedataFetch import DataProvider
from tradingbot2.TradePosition import TradePosition, TradeStatus
from tradingbot2.logger import Logger
from tradingbot2.models import TradingConfig
from tradingbot2.strategies.BaseStrategy import BaseStrategy

class TradingBot(threading.Thread):
    def __init__(self, config: TradingConfig, strategy: BaseStrategy):
        name = f"CryptoBot_{config.symbol}_{config.timeframe}"
        super().__init__(daemon=True, name=name)
        self.config = config
        self.strategy = strategy
        self.initial_capital = config.initial_balance
        self.balance = config.initial_balance
        self.stop_sig = threading.Event()
        self.logger = Logger(name=name)
        self.data_provider: DataProvider = None
        self.exception_queue = None
        self.open_position: Optional[TradePosition] = None
        self.closed_trades: List[TradePosition] = []
        
    def process_timestamp(self, row: pd.Series, previous_row: pd.Series):
        if self.open_position:
            self.open_position.update(row)
            sell, reason = self.strategy.sell_condition(self.open_position, row)
            if sell:
                self.sell_asset(row, reason)
        else:
            buy, reason, stop_loss = self.strategy.buy_condition(row, previous_row)
            if buy:
                self.buy_asset(row, reason, stop_loss)

    def buy_asset(self, row: pd.Series, reason: str, stop_loss: float):
        position_size = self.strategy.calculate_position_size(self.balance, row["close"], row)
        amount_invested = min(position_size, self.balance)
        self.balance -= amount_invested
        amount_bought = amount_invested / (row["close"] * (1 + self.config.fee_rate))
        self.open_position = TradePosition(
            symbol=self.config.symbol,
            open_time=pd.to_datetime(row['timestamp'], unit='ms'),
            open_price=row["close"],
            amount_invested=amount_invested,
            amount_bought=amount_bought,
            highest_since_purchase=row["close"],
            buy_reason=reason,
            fee_rate=self.config.fee_rate,
            stop_loss=stop_loss,
            status=TradeStatus.OPEN
        )
        self.logger.info(f"[ {self.config.symbol} ] Date : {self.open_position.open_time} - Buying at {row['close']}")


    def sell_asset(self, row: pd.Series, reason: str):
        if not self.open_position:
            return
        
        close_time = pd.to_datetime(row['timestamp'], unit='ms')
        liquidation_value = self.open_position.close(close_time, row["close"], reason)
        self.closed_trades.append(self.open_position)
        self.balance += liquidation_value
        
        self.logger.info(
            f"[ {self.config.symbol} ] Date : {close_time} - Sell at {row['close']} for a profit of {self.open_position.profit * self.config.price_usdt_rate:.2f}usd, pnl: {100 * self.open_position.sell_return:.2f} %"
        )
        self.open_position = None
    

    def portfolio_value(self) -> float:
        if self.open_position:
            return self.balance + self.open_position.liquidation_value
        return self.balance

    def stop(self):
        if self.data_provider:
            self.data_provider.stop()
            self.data_provider.join()
        self.stop_sig.set()

    def run(self):
        if not self.data_provider:
            self.logger.error("Data fetcher not initialized")
            return

        previous_row = None
        while not self.stop_sig.is_set():
            try:
                df = self.data_provider.df
                if len(df) < 14:
                    time.sleep(1)
                    continue
                df = self.strategy.calculate_indicators(df)
                previous_row = df.iloc[-2]
                newest_row = df.iloc[-1]
                self.log_status(newest_row)
                self.process_timestamp(newest_row, previous_row)
            except Exception as e:
                self.handle_exception(e)
                return
        self.cleanup()

    def log_status(self, newest_row: pd.Series):
        portfolio_value = self.portfolio_value()
        current_profit = portfolio_value - self.initial_capital
        self.logger.info(f"Current portfolio value : {portfolio_value}")
        self.logger.info(f"In position : {self.open_position is not None}")
        self.logger.info(f"Current total profit : {current_profit}")
        self.logger.info(
            f"Current pnl : {100 * current_profit / portfolio_value:.2f} (%)"
        )
        row_str = f"Open: {newest_row['open']}, High: {newest_row['high']}, Low: {newest_row['low']}, Close: {newest_row['close']}, Volume: {newest_row['volume']}, Datetime: {self.paris_datetime(newest_row['timestamp'])}"
        self.logger.info(f"Newest data row:\n {row_str}")
        self.logger.info("-" * 80)

    def handle_exception(self, e: Exception):
        if self.data_provider:
            self.data_provider.terminated.set()
            self.data_provider.join()
        if self.exception_queue:
            self.exception_queue.put(e)
        self.logger.error(f"Crashed with exception : {e}", exc_info=True)

    def cleanup(self):
        if self.data_provider:
            self.data_provider.terminated.set()
            self.data_provider.join()
        self.logger.info("Terminated")

    def get_trades(self):
        return self.closed_trades.copy()

    @staticmethod
    def paris_datetime(timestamp: int) -> str:
        return datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')