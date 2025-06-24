# Advanced Trading Bot Framework

A comprehensive cryptocurrency trading framework with multiple components:
- **Strategy Backtesting** with ClickHouse storage
- **Genetic Algorithm Optimization** 
- **Real-time Order Book Trading**
- **Smart Money Detection Dashboard**
- **Multiple Trading Strategies**

## 🏗️ Architecture

```
src/trading_bot/
├── core/                    # Models, enums, settings
├── data/                    # ClickHouse storage & Binance client
├── strategies/              # 8+ trading strategies
├── backtesting/             # Strategy backtesting engine
├── optimization/            # Genetic algorithm optimization
├── order_book_trading/      # Real-time order book strategies
└── smart_money_detection/   # Iceberg detection & dashboard
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Poetry for dependency management
- ClickHouse (preferred) or MongoDB for data storage

### Setup
```bash
# Clone and install
git clone <repo-url>
cd trading-bot
poetry install

# Setup ClickHouse (recommended)
make run-db
# OR: docker run -d --name trading-clickhouse -p 8123:8123 clickhouse/clickhouse-server

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Test the setup
poetry run python -c "from trading_bot.data.market_data import MarketDataManager; print('✅ Setup OK')"
```

### Download Market Data
```bash
# Download data for backtesting
poetry run python scripts/download_data.py --symbol BTCUSDT --days 90

# Check data status
poetry run python scripts/show_data_status.py
```

## 📊 Usage Examples

### 1. Run a Backtest
```python
from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.strategies.ema_crossover import EMACrossoverStrategy
from trading_bot.core.models import BacktestConfig
from trading_bot.core.enums import TimeFrame
from datetime import datetime
from decimal import Decimal

# Configure backtest
config = BacktestConfig(
    symbols=["BTCUSDT"],
    timeframe=TimeFrame.FIFTEEN_MINUTES,
    since_date=datetime(2024, 1, 1),
    test_start_date=datetime(2024, 6, 1),
    initial_balance=Decimal("5000")
)

# Create and run strategy
strategy = EMACrossoverStrategy(fast_period=12, slow_period=26)
engine = BacktestEngine(config, strategy)
results = engine.run()

# Print results
for symbol, result in results.items():
    print(f"{symbol}: {result.total_return_pct:.2f}% return, {result.total_trades} trades")
```

### 2. Optimize Strategy Parameters
```python
from trading_bot.optimization.genetic import optimize_strategy
from trading_bot.strategies.vwap_statistical import VWAPStatisticalStrategy

# Define parameter space
param_space = {
    'vwap_period': {'type': 'int', 'range': (10, 30)},
    'zscore_entry_threshold': {'type': 'float', 'range': (-2.5, -1.0)},
    'stop_loss_atr': {'type': 'float', 'range': (1.5, 3.0)}
}

# Run optimization
results = optimize_strategy(
    strategy_class=VWAPStatisticalStrategy,
    symbols=["BTCUSDT", "ETHUSDT"],
    parameter_space=param_space
)

print(f"Best parameters: {results['best_parameters']}")
```

### 3. Live Order Book Trading
```bash
# Test the order book trading system
poetry run python src/trading_bot/order_book_trading/tests/quick_test.py
```

### 4. Smart Money Detection Dashboard
```bash
# Start the web dashboard
poetry run python src/trading_bot/smart_money_detection/server.py BTCUSDT

# Open browser to: http://localhost:5000
```

## 🎯 Available Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `EMACrossoverStrategy` | Simple moving average crossover | Trending markets |
| `VWAPStatisticalStrategy` | Statistical reversion around VWAP | Mean reversion |
| `RSIDivergenceStrategy` | Professional RSI divergence detection | Reversal points |
| `MultiRegimeStrategy` | Adapts to volatility regimes | All market conditions |
| `TimeBasedReversionStrategy` | Exploits intraday patterns | Scalping |
| `VWAPBounceStrategy` | Dynamic VWAP support/resistance | Institutional levels |
| `MultiFactorStrategy` | Combines 6 signal types | Comprehensive |
| `SVMSlidingWindowStrategy` | Machine learning approach | Complex patterns |

## ⚙️ Configuration

Create a `.env` file:
```bash
# Database (ClickHouse preferred)
DATABASE_TYPE=clickhouse
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=8123
CLICKHOUSE_USERNAME=default
CLICKHOUSE_PASSWORD=
CLICKHOUSE_DATABASE=trading_bot

# Optional: Binance API (for live data)
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

## 🔧 Development

### Code Quality
```bash
# Format and lint
make fix-all

# Or manually:
poetry run ruff check src/ --fix
poetry run isort src/
```

### Testing
```bash
# Run tests
poetry run pytest

# Test specific component
poetry run python -m pytest tests/strategies/
```

## 📁 Key Components

### Core Framework
- `src/trading_bot/core/models.py` - Data models (Position, MarketData, etc.)
- `src/trading_bot/data/market_data.py` - Market data management
- `src/trading_bot/backtesting/engine.py` - Backtesting engine

### Trading Strategies
- `src/trading_bot/strategies/base.py` - Base strategy interface
- `src/trading_bot/strategies/*.py` - Individual strategy implementations

### Real-time Trading
- `src/trading_bot/order_book_trading/` - Live order book analysis
- `src/trading_bot/smart_money_detection/` - Iceberg detection & dashboard

### Optimization
- `src/trading_bot/optimization/genetic.py` - Genetic algorithm optimization

## 🚨 Common Issues & Solutions

### ClickHouse Connection Issues
```bash
# Check if ClickHouse is running
docker ps | grep clickhouse

# Restart ClickHouse
docker restart trading-clickhouse

# Check logs
docker logs trading-clickhouse
```

### No Market Data
```bash
# Download data first
poetry run python scripts/download_data.py --popular --days 30

# Verify data
poetry run python scripts/show_data_status.py
```

### Import Errors
```bash
# Reinstall dependencies
poetry install --no-cache

# Check Python path
poetry run python -c "import trading_bot; print('OK')"
```

## 📈 Performance Notes

- **ClickHouse vs MongoDB**: ClickHouse is ~10x faster for time-series queries
- **Timeframes**: 15m provides good balance of detail vs. speed
- **Memory**: Large datasets may require 8GB+ RAM for optimization
- **Parallelization**: Genetic optimization uses multiple cores

## 🔮 Roadmap

- [ ] Live trading execution (paper trading first)
- [ ] Additional ML strategies
- [ ] Portfolio-level backtesting
- [ ] Risk management modules
- [ ] More sophisticated order book strategies

## ⚠️ Disclaimers

- **NOT FINANCIAL ADVICE**: This is educational/research software
- **USE AT YOUR OWN RISK**: No warranty provided
- **PAPER TRADING FIRST**: Test thoroughly before risking real money
- **MARKET RISKS**: Past performance doesn't predict future results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Follow the existing code style
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details.
