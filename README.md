# Trading Bot Framework

A clean, modular cryptocurrency trading strategy framework with MongoDB storage, backtesting engine, and genetic optimization.

## 🏗️ Architecture Overview

```
src/trading_bot/
├── core/           # Core models, enums, settings
├── data/           # Market data management & MongoDB storage
├── strategies/     # Trading strategy implementations
├── backtesting/    # Strategy backtesting engine
└── optimization/   # Genetic algorithm optimization
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Poetry for dependency management
- MongoDB (for data storage)

### Setup
```bash
# Clone and install
git clone <repo-url>
cd trading-bot
poetry install

# Start MongoDB (if using Docker)
docker run --name trading-mongo -p 27017:27017 -e MONGO_INITDB_ROOT_USERNAME=admin -e MONGO_INITDB_ROOT_PASSWORD=password -d mongo

# Download market data
poetry run python scripts/download_data.py --symbol BTCUSDT --days 90

# Run a backtest
poetry run python -c "
from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.strategies.ema_crossover import EMACrossoverStrategy
from trading_bot.core.models import BacktestConfig
from trading_bot.core.enums import TimeFrame
from datetime import datetime
from decimal import Decimal

config = BacktestConfig(
    symbols=['BTCUSDT'],
    timeframe=TimeFrame.FIFTEEN_MINUTES,
    since_date=datetime(2024, 1, 1),
    test_start_date=datetime(2024, 6, 1),
    initial_balance=Decimal('5000')
)

strategy = EMACrossoverStrategy(fast_period=12, slow_period=26)
engine = BacktestEngine(config, strategy)
results = engine.run()
"
```

## 💡 Core Concepts

### Strategy Interface
All strategies inherit from `BaseStrategy` and implement:
- `buy_condition()` - When to enter trades
- `sell_condition()` - When to exit trades  
- `calculate_position_size()` - Position sizing
- `calculate_indicators()` - Technical indicators

### Data Flow
1. **Market Data** → ClickHouse via `MarketDataManager`
2. **Strategies** → Calculate indicators and signals
3. **Backtest Engine** → Simulates trading with historical data
4. **Results** → Performance metrics and trade analysis

### Available Strategies
- **EMACrossoverStrategy** - Simple moving average crossover
- **MultiRegimeStrategy** - Adapts to volatility regimes
- **VWAPStatisticalStrategy** - Statistical mean reversion around VWAP

## 🔧 Configuration

Environment variables or `.env` file:
```bash
MONGODB_URL=mongodb://admin:password@localhost:27017/trading_bot?authSource=admin
BINANCE_API_KEY=your_api_key        # Optional, for live data
BINANCE_API_SECRET=your_api_secret  # Optional, for live data
```

## 📊 Usage Examples

### Download Data
```bash
# Single symbol
poetry run python scripts/download_data.py --symbol BTCUSDT --days 30

# Multiple symbols  
poetry run python scripts/download_data.py --symbols BTCUSDT ETHUSDT --days 30

# Popular symbols
poetry run python scripts/download_data.py --popular --days 30
```

### Run Backtests
```python
from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.strategies.ema_crossover import EMACrossoverStrategy
from trading_bot.core.models import BacktestConfig

# Configure backtest
config = BacktestConfig(
    symbols=["BTCUSDT"],
    timeframe=TimeFrame.FIFTEEN_MINUTES,
    since_date=datetime(2024, 1, 1),
    test_start_date=datetime(2024, 6, 1),
    initial_balance=Decimal("5000")
)

# Create strategy
strategy = EMACrossoverStrategy(
    fast_period=12,
    slow_period=26,
    stop_loss_pct=0.02,
    take_profit_pct=0.04
)

# Run backtest
engine = BacktestEngine(config, strategy)
results = engine.run()

# Analyze results
for symbol, result in results.items():
    print(f"{symbol}: {result.total_return_pct:.2f}% return")
    print(f"Win Rate: {result.win_rate:.1%}")
    print(f"Total Trades: {result.total_trades}")
```

### Optimize Strategy
```python
from trading_bot.optimization.genetic import GeneticOptimizer

# Define parameter space
param_space = {
    'fast_period': (5, 20),
    'slow_period': (20, 50),
    'stop_loss_pct': (0.01, 0.05),
    'take_profit_pct': (0.02, 0.08)
}

# Run optimization
optimizer = GeneticOptimizer(
    strategy_class=EMACrossoverStrategy,
    config=config,
    param_space=param_space
)

best_params = optimizer.optimize(generations=20, population_size=50)
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `src/trading_bot/core/models.py` | Core data models (Position, MarketData, etc.) |
| `src/trading_bot/core/enums.py` | Enums for timeframes, trade status, etc. |
| `src/trading_bot/data/market_data.py` | Market data management |
| `src/trading_bot/backtesting/engine.py` | Main backtesting engine |
| `src/trading_bot/strategies/base.py` | Base strategy interface |
| `scripts/download_data.py` | Download market data |
| `scripts/show_data_status.py` | Check database status |

## 🧪 Development

### Code Quality
```bash
# Lint and format
poetry run ruff check src/ --fix
poetry run isort src/
poetry run black src/

# Type checking
poetry run mypy src/
```

### Testing
```bash
# Run tests
poetry run pytest

# Test specific strategy
poetry run python -m pytest tests/test_strategies.py
```

## 🔍 Troubleshooting

### Common Issues
1. **MongoDB Connection**: Check `MONGODB_URL` environment variable
2. **No Data**: Run `download_data.py` first
3. **Import Errors**: Run `poetry install` to ensure dependencies

### Debug Commands
```bash
# Check database status
poetry run python scripts/show_data_status.py

# Verify MongoDB connection
poetry run python -c "from trading_bot.data.storage import MongoStorage; print('MongoDB OK')"
```

## 📈 Performance Tips

1. **Data Management**: Download data once, reuse for multiple backtests
2. **Timeframes**: 15m timeframe balances detail vs. speed
3. **Optimization**: Start with small parameter spaces, expand gradually
4. **Memory**: Large datasets may require pagination for optimization

## 🤝 Contributing

1. Follow the existing code style (ruff + black)
2. Add tests for new strategies
3. Update documentation for new features
4. Use meaningful commit messages

## 📄 License

[MIT]
