#!/usr/bin/env python3
"""
Complete example of strategy optimization and analysis workflow
Shows how to use the genetic optimizer and analysis modules together
"""

import sys
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from trading_bot.optimization.genetic import GeneticOptimizer, optimize_strategy
from trading_bot.backtesting.analyzer import analyze_backtest_results, quick_analysis
from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.strategies.ema_crossover import EMACrossoverStrategy
from trading_bot.strategies.time_based_reversion import TimeBasedReversionStrategy
from trading_bot.core.models import BacktestConfig
from trading_bot.core.enums import TimeFrame


def optimize_ema_crossover_example():
    """Example: Optimize EMA Crossover strategy"""
    print("🧬 OPTIMIZING EMA CROSSOVER STRATEGY")
    print("=" * 50)
    
    # Define parameter space for optimization
    parameter_space = {
        'fast_period': {
            'type': 'int',
            'range': (5, 20)
        },
        'slow_period': {
            'type': 'int', 
            'range': (20, 50)
        },
        'stop_loss_pct': {
            'type': 'float',
            'range': (0.01, 0.05)
        },
        'take_profit_pct': {
            'type': 'float',
            'range': (0.02, 0.08)
        },
        'position_size_pct': {
            'type': 'float',
            'range': (0.05, 0.15)
        }
    }
    
    # Optimization settings
    optimization_settings = {
        'population_size': 30,
        'generations': 10,  # Reduced for demo
        'crossover_prob': 0.7,
        'mutation_prob': 0.2,
        'tournament_size': 3,
        'evaluation': {
            'num_evaluations': 2,
            'evaluation_days': 45,
            'max_days_back': 150,
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
    
    # Run optimization
    symbols = ["BTCUSDT", "ETHUSDT"]
    
    results = optimize_strategy(
        strategy_class=EMACrossoverStrategy,
        symbols=symbols,
        parameter_space=parameter_space,
        timeframe=TimeFrame.FIFTEEN_MINUTES,
        optimization_settings=optimization_settings
    )
    
    print(f"🎯 OPTIMIZATION RESULTS:")
    print(f"Best Fitness Score: {results['best_fitness']:.4f}")
    print(f"Best Parameters:")
    for param, value in results['best_parameters'].items():
        print(f"  {param}: {value}")
    
    # Test optimized strategy
    print(f"\n🧪 TESTING OPTIMIZED STRATEGY")
    optimized_strategy = EMACrossoverStrategy(**results['best_parameters'])
    
    # Create test configuration
    test_config = BacktestConfig(
        symbols=symbols,
        timeframe=TimeFrame.FIFTEEN_MINUTES,
        since_date=datetime.now() - timedelta(days=180),
        test_start_date=datetime.now() - timedelta(days=90),
        test_end_date=datetime.now() - timedelta(days=7),
        initial_balance=Decimal("10000"),
        fee_rate=Decimal("0.001")
    )
    
    # Run backtest with optimized parameters
    engine = BacktestEngine(test_config, optimized_strategy)
    backtest_results = engine.run()
    
    # Analyze results
    analysis = analyze_backtest_results(backtest_results, "Optimized_EMA_Crossover")
    
    return results, backtest_results, analysis


def optimize_time_based_strategy_example():
    """Example: Optimize Time-Based Reversion strategy"""
    print(f"\n🧬 OPTIMIZING TIME-BASED REVERSION STRATEGY")
    print("=" * 55)
    
    # More complex parameter space
    parameter_space = {
        'short_ma_period': {
            'type': 'int',
            'range': (5, 15)
        },
        'max_distance_from_ma': {
            'type': 'float',
            'range': (0.005, 0.025)
        },
        'rsi_period': {
            'type': 'int',
            'range': (5, 14)
        },
        'rsi_oversold': {
            'type': 'int',
            'range': (20, 35)
        },
        'max_velocity': {
            'type': 'float',
            'range': (0.02, 0.08)
        },
        'min_volume_ratio': {
            'type': 'float',
            'range': (0.8, 2.0)
        },
        'stop_loss_atr': {
            'type': 'float',
            'range': (1.5, 3.0)
        },
        'take_profit_atr': {
            'type': 'float',
            'range': (2.0, 4.0)
        },
        'position_size_pct': {
            'type': 'float',
            'range': (0.01, 0.04)
        }
    }
    
    # Create base config for volatile cryptos
    base_config = BacktestConfig(
        symbols=["PEPEUSDT", "SOLUSDT"],  # Volatile cryptos
        timeframe=TimeFrame.FIVE_MINUTES,  # Faster timeframe
        since_date=datetime.now() - timedelta(days=180),
        test_start_date=datetime.now() - timedelta(days=90),
        test_end_date=datetime.now() - timedelta(days=7),
        initial_balance=Decimal("5000"),
        fee_rate=Decimal("0.001")
    )
    
    # Custom evaluation settings for volatile cryptos
    evaluation_settings = {
        'parameter_space': parameter_space,
        'symbols': base_config.symbols,
        'num_evaluations': 3,
        'evaluation_days': 30,
        'max_days_back': 120,
        'fitness': {
            'weights': {
                'return': 0.5,  # Higher weight on returns for volatile assets
                'sharpe': 0.25,
                'drawdown': 0.15,
                'trade_count': 0.1
            }
        }
    }
    
    # Create optimizer
    optimizer = GeneticOptimizer(
        strategy_class=TimeBasedReversionStrategy,
        base_config=base_config,
        parameter_space=parameter_space,
        optimization_settings={
            'population_size': 40,
            'generations': 15,
            'crossover_prob': 0.8,
            'mutation_prob': 0.3,
            'tournament_size': 4,
            'evaluation': evaluation_settings['num_evaluations']
        }
    )
    
    # Run optimization
    results = optimizer.optimize(verbose=True)
    
    # Save results
    optimizer.save_results(results)
    
    print(f"\n🎯 OPTIMIZATION RESULTS:")
    print(f"Best Fitness: {results['best_fitness']:.4f}")
    print(f"Optimization Time: {results['optimization_time_seconds']:.1f} seconds")
    
    print(f"\nOptimal Parameters:")
    for param, value in results['best_parameters'].items():
        print(f"  {param}: {value}")
    
    # Test the optimized strategy
    print(f"\n🧪 VALIDATING OPTIMIZED STRATEGY")
    
    optimized_strategy = TimeBasedReversionStrategy(**results['best_parameters'])
    
    # Test on different time period
    validation_config = BacktestConfig(
        symbols=["PEPEUSDT", "SOLUSDT", "DOGEUSDT"],  # Add another volatile crypto
        timeframe=TimeFrame.FIVE_MINUTES,
        since_date=datetime.now() - timedelta(days=120),
        test_start_date=datetime.now() - timedelta(days=60),
        test_end_date=datetime.now() - timedelta(days=7),
        initial_balance=Decimal("5000"),
        fee_rate=Decimal("0.001")
    )
    
    engine = BacktestEngine(validation_config, optimized_strategy)
    validation_results = engine.run()
    
    # Comprehensive analysis
    analysis = analyze_backtest_results(validation_results, "Optimized_TimeBasedReversion")
    
    return results, validation_results, analysis


def compare_strategies():
    """Compare multiple strategies with their optimized parameters"""
    print(f"\n📊 STRATEGY COMPARISON")
    print("=" * 30)
    
    # Default EMA Crossover
    default_ema = EMACrossoverStrategy()
    
    # Optimized EMA Crossover (example parameters)
    optimized_ema = EMACrossoverStrategy(
        fast_period=9,
        slow_period=23,
        stop_loss_pct=0.022,
        take_profit_pct=0.045,
        position_size_pct=0.08
    )
    
    # Optimized Time-Based (example parameters)
    optimized_time_based = TimeBasedReversionStrategy(
        short_ma_period=7,
        max_distance_from_ma=0.012,
        rsi_period=8,
        rsi_oversold=25,
        max_velocity=0.05,
        min_volume_ratio=1.2,
        stop_loss_atr=2.2,
        take_profit_atr=3.5,
        position_size_pct=0.025
    )
    
    strategies = {
        "Default_EMA": default_ema,
        "Optimized_EMA": optimized_ema,
        "Optimized_TimeBased": optimized_time_based
    }
    
    # Test configuration
    test_config = BacktestConfig(
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframe=TimeFrame.FIFTEEN_MINUTES,
        since_date=datetime.now() - timedelta(days=120),
        test_start_date=datetime.now() - timedelta(days=60),
        test_end_date=datetime.now() - timedelta(days=7),
        initial_balance=Decimal("10000"),
        fee_rate=Decimal("0.001")
    )
    
    all_results = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"\n🧪 Testing {strategy_name}...")
        
        engine = BacktestEngine(test_config, strategy)
        results = engine.run()
        all_results[strategy_name] = results
        
        # Quick analysis
        quick_analysis(results, strategy_name)
    
    # Portfolio comparison
    print(f"\n🏆 STRATEGY RANKING")
    print("-" * 40)
    
    strategy_scores = {}
    for strategy_name, results in all_results.items():
        total_return = sum(result.total_return_pct for result in results.values())
        avg_sharpe = sum(result.sharpe_ratio for result in results.values()) / len(results)
        avg_win_rate = sum(result.win_rate for result in results.values()) / len(results)
        
        # Composite score
        score = total_return + avg_sharpe * 5 + avg_win_rate * 10
        strategy_scores[strategy_name] = {
            'score': score,
            'total_return': total_return,
            'avg_sharpe': avg_sharpe,
            'avg_win_rate': avg_win_rate
        }
    
    # Rank strategies
    ranked = sorted(strategy_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    for i, (strategy, metrics) in enumerate(ranked, 1):
        print(f"{i}. {strategy}")
        print(f"   Score: {metrics['score']:.2f}")
        print(f"   Return: {metrics['total_return']:.2f}%")
        print(f"   Sharpe: {metrics['avg_sharpe']:.2f}")
        print(f"   Win Rate: {metrics['avg_win_rate']:.1%}")
        print()


def main():
    """Main optimization and analysis workflow"""
    print("🚀 TRADING STRATEGY OPTIMIZATION & ANALYSIS")
    print("=" * 60)
    
    try:
        # 1. Optimize EMA Crossover strategy
        ema_opt_results, ema_backtest, ema_analysis = optimize_ema_crossover_example()
        
        # 2. Optimize Time-Based strategy  
        time_opt_results, time_backtest, time_analysis = optimize_time_based_strategy_example()
        
        # 3. Compare strategies
        compare_strategies()
        
        print(f"\n✅ OPTIMIZATION & ANALYSIS COMPLETE!")
        print(f"📁 Results saved to:")
        print(f"   • optimization_results/")
        print(f"   • analysis_reports/")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Review saved optimization results")
        print(f"   2. Test best parameters on live/paper trading")
        print(f"   3. Consider ensemble strategies")
        print(f"   4. Implement walk-forward optimization")
        
    except Exception as e:
        print(f"❌ Error in optimization workflow: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()