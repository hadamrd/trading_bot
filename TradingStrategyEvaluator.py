import random
from datetime import datetime, timedelta
from statistics import mean
from typing import List, Type

from tradingbot2.backtester import Backtester
from tradingbot2.models import BacktestConfig
from tradingbot2.strategies.BaseStrategy import BaseStrategy

class TradingStrategyEvaluator:
    def __init__(self, config, symbols_data, strategy_class: Type[BaseStrategy]):
        self.config = config
        self.symbols_data = symbols_data
        self.strategy_class: Type[BaseStrategy] = strategy_class
        self.backtest_configs: List[BacktestConfig] = self._create_backtest_configs()

    def _create_backtest_configs(self):
        backtest_configs = []
        for symbol in self.config['backtest_config']['symbols']:
            base_config = self.config['backtest_config'].copy()
            base_config['symbols'] = [symbol]
            backtest_configs.append(BacktestConfig(**base_config))
        return backtest_configs

    def _get_random_period(self, start_date, end_date, period_days):
        date_range = (end_date - start_date).days - period_days
        if date_range < 0:
            raise ValueError(f"Not enough data for a {period_days}-day period")
        random_start = start_date + timedelta(days=random.randint(0, date_range))
        random_end = random_start + timedelta(days=period_days)
        return random_start, random_end

    def evaluate(self, individual):
        params = dict(zip(self.config['individuals_genes'].keys(), individual))
        
        # Convert boolean parameters
        for key, value in params.items():
            if self.config['individuals_genes'][key]['type'] == 'boolean':
                params[key] = bool(value)
        
        strategy = self.strategy_class(**params)
        
        fitness_scores = []
        eval_period_days = self.config['eval_config']['eval_period_days']
        num_symbols = self.config['eval_config']['num_symbols']

        # Randomly select symbols for this evaluation
        eval_configs = random.sample(self.backtest_configs, num_symbols)

        for config in eval_configs:
            start_date = datetime.strptime(config.since_date, "%Y-%m-%d")
            end_date = datetime.now()
            test_start, test_end = self._get_random_period(start_date, end_date, eval_period_days)
            
            config.test_start_date = test_start.strftime("%Y-%m-%d")
            config.test_end_date = test_end.strftime("%Y-%m-%d")
            
            backtester = Backtester(config, strategy)
            results = backtester.run_on_prepared_data(self.symbols_data[config.symbols[0]])
            
            # Calculate average fitness across all symbols
            symbol_fitnesses = []
            for symbol, result in results.items():
                total_return = result.total_profit / config.initial_balance
                fitness = total_return * (1 + result.sortino_ratio)
                symbol_fitnesses.append(fitness)
            
            average_symbol_fitness = mean(symbol_fitnesses)
            fitness_scores.append(average_symbol_fitness)
        
        average_fitness = mean(fitness_scores)
        return (average_fitness,)