from dataclasses import dataclass
import json
import logging
from typing import Dict, Any, Type
from datetime import datetime
from pathlib import Path

from tradingbot2.genetic_algo import GeneticAlgorithm
from tradingbot2.TradingStrategyEvaluator import TradingStrategyEvaluator
from tradingbot2.backtest_data_manager import DataManager
from tradingbot2.models import BacktestConfig, OptimizationResult
from tradingbot2.strategies.BaseStrategy import BaseStrategy

logger = logging.getLogger(__name__)

class GeneticOptimizer:
    """
    Handles the genetic optimization process for trading strategies.
    Encapsulates data preparation, optimization, and results handling.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize optimizer with configuration file path.
        
        Args:
            config_path: Path to the configuration JSON file
        """
        self.config = self._load_config(config_path)
        self.strategy_class = self._get_strategy_class()
        self._validate_config()
        
    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Load and parse configuration file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            raise
            
    def _get_strategy_class(self) -> Type[BaseStrategy]:
        """Get strategy class from configuration"""
        strategy_name = self.config.get('strategy', {}).get('class')
        if not strategy_name:
            raise ValueError("Strategy class not specified in config")
            
        # Import strategy dynamically
        try:
            module_path, class_name = strategy_name.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except Exception as e:
            logger.error(f"Failed to load strategy class {strategy_name}: {e}")
            raise
            
    def _validate_config(self):
        """Validate configuration structure and values"""
        required_keys = {'backtest_config', 'optimization', 'strategy'}
        if not all(key in self.config for key in required_keys):
            raise ValueError(f"Missing required configuration keys: {required_keys}")
        
        # Create BacktestConfig object to validate backtest configuration
        try:
            BacktestConfig(**self.config['backtest_config'])
        except Exception as e:
            logger.error(f"Invalid backtest configuration: {e}")
            raise

    def _prepare_data(self) -> Dict[str, Any]:
        """
        Prepare and cache data for all symbols with indicators.
        Hidden from external use as it's part of the optimization process.
        """
        backtest_config = BacktestConfig(**self.config['backtest_config'])
        
        try:
            return DataManager.prepare_strategy_data(
                backtest_config=backtest_config,
                strategy_class=self.strategy_class,
                strategy_params=self.config['strategy'].get('default_params', {})
            )
        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            raise

    def _save_results(self, result: OptimizationResult):
        """Save optimization results to file"""
        output_dir = Path("optimization_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"optimization_{self.strategy_class.__name__}_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(result.dict(), f, indent=2)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise

    def run(self) -> OptimizationResult:
        """
        Run the genetic optimization process.
        
        Returns:
            OptimizationResult containing the best parameters and optimization metrics
        """
        logger.info("Starting genetic optimization process")
        start_time = datetime.now()
        
        try:
            # Prepare data once for all evaluations
            logger.info("Preparing market data...")
            symbols_data = self._prepare_data()
            
            # Initialize evaluator and genetic algorithm
            logger.info("Initializing optimization components...")
            evaluator = TradingStrategyEvaluator(
                config=self.config,
                symbols_data=symbols_data,
                strategy_class=self.strategy_class
            )
            
            ga = GeneticAlgorithm(
                config=self.config['optimization'],
                evaluator=evaluator
            )
            
            # Run optimization
            logger.info("Running genetic algorithm...")
            best_params, logbook = ga.run()
            
            # Create result object
            run_duration = (datetime.now() - start_time).total_seconds()
            result = OptimizationResult(
                strategy_name=self.strategy_class.__name__,
                best_parameters=best_params,
                indicator_parameters=self.strategy_class.get_indicator_params(best_params),
                best_fitness=logbook.select("max")[-1],
                run_info={
                    "datetime": datetime.now().isoformat(),
                    "duration_seconds": run_duration,
                    "config": self.config
                },
                evolution={
                    "generations": logbook.select("gen"),
                    "min_fitness": logbook.select("min"),
                    "avg_fitness": logbook.select("avg"),
                    "max_fitness": logbook.select("max")
                }
            )
            
            # Save results
            self._save_results(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
