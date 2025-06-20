from .src.trading_bot.core.models import BacktestResult, BacktestConfig, TradingConfig
from .backtest_data_manager import DataManager
from .results_analyzer import ResultsAnalyzer
from .cryptobot import TradingBot
from .logger import Logger
from .strategies.BaseStrategy import BaseStrategy
import pandas as pd

class Backtester:
    def __init__(self, config: BacktestConfig, strategy: BaseStrategy):
        self.config = config
        self.strategy = strategy
        self.logger = Logger(name="Backtester")

    def run(self) -> dict[str, BacktestResult]:
        results = {}
        for symbol in self.config.symbols:
            self.logger.info(f"Backtesting {symbol}")
            full_df, test_df = DataManager.get_symbol_data(self.config, symbol, self.strategy)
            
            trading_config = TradingConfig(
                symbol=symbol,
                timeframe=self.config.timeframe,
                price_usdt_rate=self.config.price_usdt_rate,
                fee_rate=self.config.fee_rate,
                initial_balance=self.config.initial_balance
            )
            
            bot = TradingBot(config=trading_config, strategy=self.strategy)
            
            result = self.backtest_symbol(bot, test_df)
            results[symbol] = result
        
        return results

    def fetch_data(self):
        return {
            symbol: DataManager.fetch_data(symbol, self.config.timeframe, self.config.since_date, self.config.test_end_date)
                for symbol in self.config.symbols
        }

    def run_on_prepared_data(self, market_data) -> dict[str, BacktestResult]:
        results = {}
        for symbol, df in market_data.items():
            test_df = DataManager.prepare_data(df, self.config, self.strategy)
            trading_config = TradingConfig(
                symbol=symbol,
                timeframe=self.config.timeframe,
                price_usdt_rate=self.config.price_usdt_rate,
                fee_rate=self.config.fee_rate,
                initial_balance=self.config.initial_balance
            )
            bot = TradingBot(config=trading_config, strategy=self.strategy)
            result = self.backtest_symbol(bot, test_df)
            results[symbol] = result
        return results

    def backtest_symbol(self, bot: TradingBot, df: pd.DataFrame) -> BacktestResult:
        previous_row = None
        for _, row in df.iterrows():
            if previous_row is not None:
                bot.process_timestamp(row, previous_row)
            previous_row = row
        
        if bot.open_position:
            self.logger.info("Closing trailing open position")
            bot.sell_asset(df.iloc[-1], "End of backtest")
        
        trades = bot.get_trades()
        return ResultsAnalyzer.generate_backtest_result(bot.config.symbol, trades)