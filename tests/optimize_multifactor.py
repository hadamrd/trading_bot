#!/usr/bin/env python3
"""
Genetic optimization for Multi-Factor Strategy
Find the best parameters before testing
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal

sys.path.append(str(Path(__file__).parent.parent / "src"))

from trading_bot.optimization.genetic import GeneticOptimizer
from trading_bot.strategies.multi_factor import MultiFactorStrategy
from trading_bot.core.models import BacktestConfig
from trading_bot.core.enums import TimeFrame


def optimize_multi_factor():
    """Optimize Multi-Factor Strategy parameters"""
    
    print("🧬 OPTIMIZING MULTI-FACTOR STRATEGY")
    print("=" * 50)
    
    # Base configuration for optimization
    base_config = BacktestConfig(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "PEPEUSDT"],
        timeframe=TimeFrame.FIVE_MINUTES,
        since_date=datetime(2024, 6, 1),    # Longer history for optimization
        test_start_date=datetime(2024, 9, 1),
        test_end_date=datetime(2024, 12, 31),  # Use Sep-Dec for optimization
        initial_balance=Decimal("10000"),
        fee_rate=Decimal("0.001")
    )
    
    print(f"Optimization Period: {base_config.test_start_date.date()} to {base_config.test_end_date.date()}")
    print(f"Symbols: {', '.join(base_config.symbols)}")
    
    # Define parameter space for optimization
    parameter_space = {
        # Core EMA parameters
        'ema_short_period': {
            'type': 'int',
            'range': (5, 15)
        },
        'ema_long_period': {
            'type': 'int', 
            'range': (15, 30)
        },
        
        # RSI parameters
        'rsi_period': {
            'type': 'int',
            'range': (10, 20)
        },
        
        # Risk management
        'risk_per_trade': {
            'type': 'float',
            'range': (0.005, 0.025)
        },
        'take_profit_factor': {
            'type': 'float',
            'range': (1.0, 3.0)
        },
        'stop_loss_factor': {
            'type': 'float',
            'range': (0.8, 1.5)
        },
        
        # Signal-specific RSI thresholds
        'short_cross_rsi_threshold': {
            'type': 'float',
            'range': (40, 70)
        },
        'vwap_bounce_rsi_threshold': {
            'type': 'float',
            'range': (50, 80)
        },
        'ema_golden_cross_rsi_threshold': {
            'type': 'float',
            'range': (30, 60)
        },
        
        # Signal-specific ATR factors
        'short_cross_atr_factor': {
            'type': 'float',
            'range': (0.5, 1.5)
        },
        'vwap_bounce_atr_factor': {
            'type': 'float',
            'range': (0.8, 1.5)
        },
        'ema_golden_cross_atr_factor': {
            'type': 'float',
            'range': (0.8, 1.5)
        },
        
        # Signal selection (which signals to use)
        'use_short_cross': {
            'type': 'bool',
            'range': (0, 1)
        },
        'use_vwap_bounce': {
            'type': 'bool',
            'range': (0, 1)
        },
        'use_ema_golden_cross': {
            'type': 'bool',
            'range': (0, 1)
        }
    }
    
    print(f"Optimizing {len(parameter_space)} parameters...")
    print("Key parameters:")
    for param in ['ema_short_period', 'ema_long_period', 'risk_per_trade', 
                  'use_short_cross', 'use_vwap_bounce', 'use_ema_golden_cross']:
        if param in parameter_space:
            print(f"  • {param}: {parameter_space[param]['range']}")
    
    # Optimization settings
    optimization_settings = {
        'population_size': 50,     # Larger population
        'generations': 20,         # More generations  
        'crossover_prob': 0.8,
        'mutation_prob': 0.3,
        'tournament_size': 5,
        'evaluation': {
            'num_evaluations': 3,      # More evaluations per individual
            'evaluation_days': 45,     # Longer evaluation periods
            'max_days_back': 120,
            'fitness': {
                'weights': {
                    'return': 0.5,         # High weight on returns
                    'sharpe': 0.25,        # Risk-adjusted returns
                    'drawdown': 0.15,      # Penalty for high drawdown
                    'trade_count': 0.1     # Need sufficient trades
                }
            }
        }
    }
    
    print(f"\nGenetic Algorithm Settings:")
    print(f"  Population: {optimization_settings['population_size']}")
    print(f"  Generations: {optimization_settings['generations']}")
    print(f"  Evaluations per individual: {optimization_settings['evaluation']['num_evaluations']}")
    
    # Create optimizer
    optimizer = GeneticOptimizer(
        strategy_class=MultiFactorStrategy,
        base_config=base_config,
        parameter_space=parameter_space,
        optimization_settings=optimization_settings
    )
    
    # Run optimization
    print(f"\n🚀 Starting optimization...")
    print(f"This will take approximately {optimization_settings['population_size'] * optimization_settings['generations'] * optimization_settings['evaluation']['num_evaluations'] / 60:.0f} minutes")
    
    results = optimizer.optimize(verbose=True)
    
    # Save results
    results_file = optimizer.save_results(results)
    
    # Display results
    print(f"\n🎯 OPTIMIZATION COMPLETE!")
    print(f"=" * 40)
    print(f"Best Fitness Score: {results['best_fitness']:.4f}")
    print(f"Optimization Time: {results['optimization_time_seconds']:.1f} seconds")
    print(f"Results saved to: {results_file}")
    
    print(f"\n🏆 BEST PARAMETERS:")
    best_params = results['best_parameters']
    for param, value in best_params.items():
        if isinstance(value, bool):
            print(f"  {param}: {'✅' if value else '❌'}")
        elif isinstance(value, float):
            print(f"  {param}: {value:.4f}")
        else:
            print(f"  {param}: {value}")
    
    # Show signal selection
    print(f"\n📊 OPTIMIZED SIGNAL SELECTION:")
    signals = {
        'Short Cross': best_params.get('use_short_cross', True),
        'Long Cross': best_params.get('use_long_cross', True), 
        'Short Bounce': best_params.get('use_short_bounce', True),
        'Long Bounce': best_params.get('use_long_bounce', True),
        'VWAP Bounce': best_params.get('use_vwap_bounce', True),
        'EMA Golden Cross': best_params.get('use_ema_golden_cross', True)
    }
    
    active_signals = [name for name, active in signals.items() if active]
    inactive_signals = [name for name, active in signals.items() if not active]
    
    print(f"  Active Signals ({len(active_signals)}):")
    for signal in active_signals:
        print(f"    ✅ {signal}")
    
    if inactive_signals:
        print(f"  Disabled Signals ({len(inactive_signals)}):")
        for signal in inactive_signals:
            print(f"    ❌ {signal}")
    
    # Compare to previous optimizations
    print(f"\n📈 OPTIMIZATION COMPARISON:")
    print(f"Strategy                Fitness Score")
    print(f"-" * 40)
    print(f"EMA Crossover           0.184")
    print(f"Time-Based Reversion    0.255") 
    print(f"Multi-Factor            {results['best_fitness']:.3f}")
    
    if results['best_fitness'] > 0.255:
        improvement = (results['best_fitness'] / 0.255 - 1) * 100
        print(f"🎉 Multi-Factor is {improvement:+.1f}% better than Time-Based!")
    elif results['best_fitness'] > 0.184:
        print(f"📈 Multi-Factor beats EMA Crossover")
    else:
        print(f"⚠️  Multi-Factor fitness lower than previous strategies")
    
    # Next steps
    print(f"\n🚀 NEXT STEPS:")
    print(f"1. Test optimized parameters on Jan-May 2025 (out-of-sample)")
    print(f"2. If profitable: consider paper trading")
    print(f"3. If still loses money: try different approach entirely")
    print(f"4. The optimized parameters are ready for testing!")
    
    return results


if __name__ == "__main__":
    optimize_multi_factor()