"""
Genetic Algorithm Optimizer for Trading Strategies
Integrated with the new trading bot framework
"""

import json
import logging
import random
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Type

import numpy as np
from deap import algorithms, base, creator, tools

from ..backtesting.engine import BacktestEngine
from ..core.models import BacktestConfig
from ..core.enums import TimeFrame
from ..strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class StrategyEvaluator:
    """Evaluates trading strategies for genetic optimization"""
    
    def __init__(self, 
                 base_config: BacktestConfig,
                 strategy_class: Type[BaseStrategy],
                 evaluation_settings: Dict[str, Any]):
        self.base_config = base_config
        self.strategy_class = strategy_class
        self.evaluation_settings = evaluation_settings
        self.fitness_cache = {}
        
    def evaluate(self, individual: List[float]) -> tuple[float]:
        """
        Evaluate an individual (parameter set) and return fitness score
        
        Args:
            individual: List of parameter values
            
        Returns:
            Tuple with single fitness value
        """
        # Convert individual to parameter dict
        params = self._individual_to_params(individual)
        
        # Check cache first
        params_key = str(sorted(params.items()))
        if params_key in self.fitness_cache:
            return self.fitness_cache[params_key]
        
        try:
            # Create strategy with these parameters
            strategy = self.strategy_class(**params)
            
            # Run backtests on random periods/symbols
            fitness_scores = []
            
            for _ in range(self.evaluation_settings.get('num_evaluations', 3)):
                # Create random evaluation config
                eval_config = self._create_random_eval_config()
                
                # Run backtest
                engine = BacktestEngine(eval_config, strategy)
                results = engine.run()
                
                # Calculate fitness from results
                fitness = self._calculate_fitness(results)
                fitness_scores.append(fitness)
            
            # Average fitness across evaluations
            avg_fitness = np.mean(fitness_scores)
            
            # Cache result
            self.fitness_cache[params_key] = (avg_fitness,)
            
            return (avg_fitness,)
            
        except Exception as e:
            logger.warning(f"Evaluation failed for params {params}: {e}")
            return (-1000.0,)  # Penalty for failed evaluation
    
    def _individual_to_params(self, individual: List[float]) -> Dict[str, Any]:
        """Convert DEAP individual to strategy parameters"""
        params = {}
        param_definitions = self.evaluation_settings['parameter_space']
        
        for i, (param_name, param_config) in enumerate(param_definitions.items()):
            value = individual[i]
            
            if param_config['type'] == 'int':
                params[param_name] = int(round(value))
            elif param_config['type'] == 'float':
                params[param_name] = float(value)
            elif param_config['type'] == 'bool':
                params[param_name] = bool(round(value))
            
        return params
    
    def _create_random_eval_config(self) -> BacktestConfig:
        """Create a random evaluation configuration"""
        # Random symbol from available symbols
        symbols = self.evaluation_settings.get('symbols', self.base_config.symbols)
        random_symbol = random.choice(symbols)
        
        # Random time period
        eval_days = self.evaluation_settings.get('evaluation_days', 30)
        max_days_back = self.evaluation_settings.get('max_days_back', 180)
        
        # Random start date within the range
        days_back = random.randint(eval_days, max_days_back)
        random_end = datetime.now() - timedelta(days=random.randint(1, 30))
        random_start = random_end - timedelta(days=eval_days)
        
        return BacktestConfig(
            symbols=[random_symbol],
            timeframe=self.base_config.timeframe,
            since_date=random_start - timedelta(days=50),  # Extra for indicators
            test_start_date=random_start,
            test_end_date=random_end,
            initial_balance=self.base_config.initial_balance,
            fee_rate=self.base_config.fee_rate
        )
    
    def _calculate_fitness(self, results: Dict[str, Any]) -> float:
        """Calculate fitness score from backtest results"""
        fitness_config = self.evaluation_settings.get('fitness', {})
        
        total_fitness = 0.0
        
        for symbol, result in results.items():
            # Base fitness components
            total_return = result.total_return_pct / 100.0  # Convert to decimal
            sharpe_ratio = result.sharpe_ratio
            max_drawdown = result.max_drawdown / 100.0  # Convert to decimal
            win_rate = result.win_rate
            total_trades = result.total_trades
            
            # Weighted fitness calculation
            weights = fitness_config.get('weights', {
                'return': 0.4,
                'sharpe': 0.3,
                'drawdown': 0.2,
                'trade_count': 0.1
            })
            
            # Individual components
            return_score = total_return * weights['return']
            sharpe_score = min(sharpe_ratio, 3.0) * weights['sharpe']  # Cap sharpe
            drawdown_score = max(0, (0.2 - max_drawdown)) * weights['drawdown']  # Penalty for high DD
            
            # Trade count bonus (prefer strategies that actually trade)
            if total_trades >= 5:
                trade_score = min(total_trades / 20.0, 1.0) * weights['trade_count']
            else:
                trade_score = -0.5  # Penalty for too few trades
            
            symbol_fitness = return_score + sharpe_score + drawdown_score + trade_score
            total_fitness += symbol_fitness
        
        return total_fitness / len(results) if results else -1000.0


class GeneticOptimizer:
    """
    Genetic Algorithm Optimizer for Trading Strategies
    """
    
    def __init__(self, 
                 strategy_class: Type[BaseStrategy],
                 base_config: BacktestConfig,
                 parameter_space: Dict[str, Dict[str, Any]],
                 optimization_settings: Dict[str, Any] = None):
        """
        Initialize the genetic optimizer
        
        Args:
            strategy_class: Strategy class to optimize
            base_config: Base configuration for backtesting
            parameter_space: Dictionary defining parameter ranges
            optimization_settings: GA settings
        """
        self.strategy_class = strategy_class
        self.base_config = base_config
        self.parameter_space = parameter_space
        self.settings = optimization_settings or self._default_settings()
        
        # Setup evaluator first
        evaluation_settings = {
            'parameter_space': parameter_space,
            'symbols': base_config.symbols,
            **self.settings.get('evaluation', {})
        }
        self.evaluator = StrategyEvaluator(base_config, strategy_class, evaluation_settings)
        
        # Setup DEAP toolbox after evaluator is created
        self._setup_toolbox()
    
    def _default_settings(self) -> Dict[str, Any]:
        """Default optimization settings"""
        return {
            'population_size': 50,
            'generations': 20,
            'crossover_prob': 0.7,
            'mutation_prob': 0.2,
            'tournament_size': 3,
            'elite_size': 2,
            'evaluation': {
                'num_evaluations': 2,
                'evaluation_days': 30,
                'max_days_back': 180,
                'fitness': {
                    'weights': {
                        'return': 0.4,
                        'sharpe': 0.3,
                        'drawdown': 0.2,
                        'trade_count': 0.1
                    }
                }
            }
        }
    
    def _setup_toolbox(self):
        """Setup DEAP genetic algorithm toolbox"""
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # Register parameter generators
        for param_name, param_config in self.parameter_space.items():
            if param_config['type'] == 'int':
                min_val, max_val = param_config['range']
                self.toolbox.register(f"attr_{param_name}", 
                                    random.randint, min_val, max_val)
            elif param_config['type'] == 'float':
                min_val, max_val = param_config['range']
                self.toolbox.register(f"attr_{param_name}", 
                                    random.uniform, min_val, max_val)
            elif param_config['type'] == 'bool':
                self.toolbox.register(f"attr_{param_name}", 
                                    random.randint, 0, 1)
        
        # Register individual and population
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                            [getattr(self.toolbox, f"attr_{name}") 
                             for name in self.parameter_space.keys()], n=1)
        
        self.toolbox.register("population", tools.initRepeat, 
                            list, self.toolbox.individual)
        
        # Register genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._custom_mutate, 
                            indpb=0.3)
        self.toolbox.register("select", tools.selTournament, 
                            tournsize=self.settings['tournament_size'])
        self.toolbox.register("evaluate", self.evaluator.evaluate)
    
    def _custom_mutate(self, individual, indpb):
        """Custom mutation function respecting parameter types and ranges"""
        for i, (param_name, param_config) in enumerate(self.parameter_space.items()):
            if random.random() < indpb:
                if param_config['type'] == 'int':
                    min_val, max_val = param_config['range']
                    individual[i] = random.randint(min_val, max_val)
                elif param_config['type'] == 'float':
                    min_val, max_val = param_config['range']
                    individual[i] = random.uniform(min_val, max_val)
                elif param_config['type'] == 'bool':
                    individual[i] = random.randint(0, 1)
        return (individual,)
    
    def optimize(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run genetic optimization
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting genetic optimization for {self.strategy_class.__name__}")
        logger.info(f"Population: {self.settings['population_size']}, "
                   f"Generations: {self.settings['generations']}")
        
        start_time = datetime.now()
        
        # Initialize population
        population = self.toolbox.population(n=self.settings['population_size'])
        
        # Statistics tracking
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("std", np.std)
        
        # Hall of fame (best individuals)
        hof = tools.HallOfFame(self.settings.get('elite_size', 5))
        
        # Run evolution
        final_pop, logbook = algorithms.eaSimple(
            population, self.toolbox,
            cxpb=self.settings['crossover_prob'],
            mutpb=self.settings['mutation_prob'],
            ngen=self.settings['generations'],
            stats=stats,
            halloffame=hof,
            verbose=verbose
        )
        
        # Get best individual
        best_individual = hof[0]
        best_params = self.evaluator._individual_to_params(best_individual)
        best_fitness = best_individual.fitness.values[0]
        
        # Optimization results
        duration = (datetime.now() - start_time).total_seconds()
        
        results = {
            'strategy_class': self.strategy_class.__name__,
            'best_parameters': best_params,
            'best_fitness': best_fitness,
            'optimization_time_seconds': duration,
            'generations_completed': self.settings['generations'],
            'final_population_size': len(final_pop),
            'evolution_stats': {
                'generations': logbook.select('gen'),
                'avg_fitness': logbook.select('avg'),
                'max_fitness': logbook.select('max'),
                'min_fitness': logbook.select('min'),
                'std_fitness': logbook.select('std')
            },
            'hall_of_fame': [
                {
                    'parameters': self.evaluator._individual_to_params(ind),
                    'fitness': ind.fitness.values[0]
                }
                for ind in hof
            ]
        }
        
        logger.info(f"Optimization completed in {duration:.2f} seconds")
        logger.info(f"Best fitness: {best_fitness:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save optimization results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_{self.strategy_class.__name__}_{timestamp}.json"
        
        output_dir = Path("optimization_results")
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
        return filepath


def optimize_strategy(strategy_class: Type[BaseStrategy],
                     symbols: List[str],
                     parameter_space: Dict[str, Dict[str, Any]],
                     timeframe: TimeFrame = TimeFrame.FIFTEEN_MINUTES,
                     optimization_settings: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convenience function to optimize a strategy
    
    Args:
        strategy_class: Strategy class to optimize
        symbols: List of symbols to test on
        parameter_space: Parameter ranges to optimize
        timeframe: Trading timeframe
        optimization_settings: GA settings
        
    Returns:
        Optimization results
    """
    from decimal import Decimal
    
    # Create base config
    base_config = BacktestConfig(
        symbols=symbols,
        timeframe=timeframe,
        since_date=datetime.now() - timedelta(days=365),
        test_start_date=datetime.now() - timedelta(days=180),
        test_end_date=datetime.now() - timedelta(days=30),
        initial_balance=Decimal("10000"),
        fee_rate=Decimal("0.001")
    )
    
    # Create optimizer
    optimizer = GeneticOptimizer(
        strategy_class=strategy_class,
        base_config=base_config,
        parameter_space=parameter_space,
        optimization_settings=optimization_settings
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Save results
    optimizer.save_results(results)
    
    return results