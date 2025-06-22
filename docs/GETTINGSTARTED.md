# Getting Started with Trading Bot Framework

This guide walks you through setting up the trading bot framework and running your first backtest.

## 🚀 Quick Setup (5 minutes)

### 1. Prerequisites Check
```bash
# Check Python version (3.10+ required)
python --version

# Check if Poetry is installed
poetry --version

# If not installed:
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Clone and Install
```bash
# Clone the repository
git clone <your-repo-url>
cd trading-bot

# Install dependencies
poetry install

# Enter the poetry shell
poetry shell
```

### 3. Start MongoDB (Choose one option)

#### Option A: Docker (Recommended)
```bash
# Start MongoDB with Docker
docker run --name trading-mongo \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=password \
  -d mongo

# Verify it's running
docker ps
```

#### Option B: Local MongoDB
```bash
# If you have MongoDB installed locally
mongod --dbpath /path/to/your/data/directory
```

#### Option C: MongoDB Atlas (Cloud)
```bash
# Create a .env file with your Atlas connection string
echo "MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/trading_bot" > .env
```

### 4. Verify Setup
```bash
# Check database connection
python scripts/show_data_status.py

# Should output: "No data found in database" (this is expected)
```

## 📊 Your First Backtest (10 minutes)

### Step 1: Download Market Data
```bash
# Download 30 days of Bitcoin data
python scripts/download_data.py --symbol BTCUSDT --days 30

# This will:
# - Fetch OHLCV data from Binance
# - Store it in MongoDB
# - Calculate technical indicators
# - Show progress bar
```

### Step 2: Create Your First Backtest Script

Create a file called `my_first_backtest.py`:

```python
from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.strategies.ema_crossover import EMACrossoverStrategy
from trading_bot.core.models import BacktestConfig
from trading_bot.core.enums import TimeFrame
from datetime import datetime, timedelta
from decimal import Decimal

# Configure the backtest
config = BacktestConfig(
    symbols=["BTCUSDT"],                                    # Trade Bitcoin
    timeframe=TimeFrame.FIFTEEN_MINUTES,                   # 15-minute candles
    since_date=datetime.now() - timedelta(days=30),        # Start 30 days ago
    test_start_date=datetime.now() - timedelta(days=15),   # Test last 15 days
    initial_balance=Decimal("5000"),                       # Start with $5000
    fee_rate=Decimal("0.001")                             # 0.1% trading fee
)

# Create a simple EMA crossover strategy
strategy = EMACrossoverStrategy(
    fast_period=12,          # 12-period fast EMA
    slow_period=26,          # 26-period slow EMA
    stop_loss_pct=0.02,      # 2% stop loss
    take_profit_pct=0.04,    # 4% take profit
    position_size_pct=0.1    # Use 10% of balance per trade
)

# Run the backtest
print("🚀 Running backtest...")
engine = BacktestEngine(config, strategy)
results = engine.run()

# Print results
for symbol, result in results.items():
    print(f"\n📊 Results for {symbol}:")
    print(f"   Total Return: ${result.total_return:.2f} ({result.total_return_pct:.2f}%)")
    print(f"   Total Trades: {result.total_trades}")
    print(f"   Win Rate: {result.win_rate:.1%}")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {result.max_drawdown:.2f}%")
    
    if result.total_trades > 0:
        print(f"   Average Trade: ${result.total_return / result.total_trades:.2f}")
        print(f"   Best Trade: ${result.largest_win:.2f}")
        print(f"   Worst Trade: ${result.largest_loss:.2f}")
```

### Step 3: Run Your Backtest
```bash
python my_first_backtest.py
```

Expected output:
```
🚀 Running backtest...
📊 Processing 1440 candles for BTCUSDT (test period)

📊 Results for BTCUSDT:
   Total Return: $127.50 (2.55%)
   Total Trades: 8
   Win Rate: 62.5%
   Sharpe Ratio: 1.23
   Max Drawdown: -3.45%
   Average Trade: $15.94
   Best Trade: $89.32
   Worst Trade: -$67.21
```

## 🔍 Understanding the Results

### Key Metrics Explained

- **Total Return**: Absolute profit/loss in dollars
- **Total Return %**: Percentage gain/loss on initial balance
- **Total Trades**: Number of buy/sell cycles completed
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Max Drawdown**: Largest peak-to-trough decline

### What Makes a Good Strategy?
- **Win Rate**: 55%+ is good for crypto
- **Sharpe Ratio**: >1.0 is decent, >2.0 is excellent
- **Max Drawdown**: <20% is manageable
- **Total Return**: Should beat buy-and-hold

## 🎯 Try Different Strategies

### EMA Crossover (Trend Following)
```python
# Good for trending markets
strategy = EMACrossoverStrategy(
    fast_period=9,           # Faster signals
    slow_period=21,          # Standard
    stop_loss_pct=0.025,     # 2.5% stop
    take_profit_pct=0.05     # 5% target
)
```

### Multi-Regime (Adaptive)
```python
from trading_bot.strategies.multi_regime import MultiRegimeStrategy

# Adapts to market conditions
strategy = MultiRegimeStrategy(
    volatility_threshold_high=0.7,    # High volatility threshold
    rsi_oversold=30,                  # Mean reversion signal
    ema_fast=12,                      # Momentum signal
    ema_slow=26,
    stop_loss_atr=2.0,               # ATR-based stops
    position_size_pct=0.02           # Smaller positions
)
```

### VWAP Statistical (Mean Reversion)
```python
from trading_bot.strategies.vwap_statistical import VWAPStatisticalStrategy

# Statistical mean reversion
strategy = VWAPStatisticalStrategy(
    vwap_period=20,                  # VWAP calculation period
    zscore_entry_threshold=-1.5,     # Buy when oversold
    zscore_exit_threshold=0.5,       # Sell when reverting
    min_volume_ratio=1.2,            # Volume confirmation
    position_size_pct=0.03
)
```

## 📈 Expand Your Testing

### Test Multiple Symbols
```python
config = BacktestConfig(
    symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT"],  # Test 3 symbols
    timeframe=TimeFrame.FIFTEEN_MINUTES,
    since_date=datetime.now() - timedelta(days=90),  # More data
    test_start_date=datetime.now() - timedelta(days=30),
    initial_balance=Decimal("5000")
)
```

### Test Different Timeframes
```python
# Try different timeframes
timeframes_to_test = [
    TimeFrame.FIVE_MINUTES,
    TimeFrame.FIFTEEN_MINUTES,
    TimeFrame.ONE_HOUR
]

for tf in timeframes_to_test:
    config.timeframe = tf
    results = engine.run()
    # Compare results
```

### Longer Time Periods
```bash
# Download more data first
python scripts/download_data.py --symbol BTCUSDT --days 180

# Then test longer periods
config = BacktestConfig(
    since_date=datetime.now() - timedelta(days=180),
    test_start_date=datetime.now() - timedelta(days=90),
    test_end_date=datetime.now() - timedelta(days=30)
)
```

## 🔧 Optimize Your Strategy

### Manual Parameter Testing
```python
# Test different parameter combinations
fast_periods = [9, 12, 15]
slow_periods = [21, 26, 30]

best_return = -float('inf')
best_params = None

for fast in fast_periods:
    for slow in slow_periods:
        if fast >= slow:  # Skip invalid combinations
            continue
            
        strategy = EMACrossoverStrategy(
            fast_period=fast,
            slow_period=slow
        )
        
        engine = BacktestEngine(config, strategy)
        results = engine.run()
        
        total_return = sum(r.total_return_pct for r in results.values())
        
        if total_return > best_return:
            best_return = total_return
            best_params = (fast, slow)
            
print(f"Best parameters: fast={best_params[0]}, slow={best_params[1]}")
print(f"Best return: {best_return:.2f}%")
```

### Genetic Algorithm Optimization
```python
from trading_bot.optimization.genetic import GeneticOptimizer

# Define parameter search space
param_space = {
    'fast_period': (5, 20),
    'slow_period': (15, 40),
    'stop_loss_pct': (0.01, 0.05),
    'take_profit_pct': (0.02, 0.08)
}

# Run optimization
optimizer = GeneticOptimizer(
    strategy_class=EMACrossoverStrategy,
    config=config,
    param_space=param_space
)

print("🧬 Running genetic optimization...")
best_params = optimizer.optimize(
    generations=10,      # Start small
    population_size=20   # Start small
)

print(f"Optimal parameters: {best_params}")
```

## 🗄️ Data Management

### Download Data for Multiple Symbols
```bash
# Download popular symbols
python scripts/download_data.py --popular --days 60

# Or specific symbols
python scripts/download_data.py --symbols BTCUSDT ETHUSDT BNBUSDT --days 60

# Check what you have
python scripts/show_data_status.py
```

### Environment Configuration
Create a `.env` file for configuration:
```bash
# .env file
MONGODB_URL=mongodb://admin:password@localhost:27017/trading_bot?authSource=admin
BINANCE_API_KEY=your_api_key_here        # Optional
BINANCE_API_SECRET=your_api_secret_here  # Optional
LOG_LEVEL=INFO
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### "No data found"
```bash
# Check if data exists
python scripts/show_data_status.py

# Download data if missing
python scripts/download_data.py --symbol BTCUSDT --days 30
```

#### "MongoDB connection failed"
```bash
# Check if MongoDB is running
docker ps | grep mongo

# Restart if needed
docker restart trading-mongo

# Test connection
python -c "from trading_bot.data.storage import MongoStorage; print('MongoDB OK')"
```

#### "Not enough data for indicators"
```bash
# Download more historical data
python scripts/download_data.py --symbol BTCUSDT --days 90

# Or use shorter indicator periods
strategy = EMACrossoverStrategy(fast_period=5, slow_period=10)
```

#### "ImportError"
```bash
# Reinstall dependencies
poetry install

# Check you're in poetry shell
poetry shell
```

### Performance Issues

#### Slow backtests
- Use shorter time periods for testing
- Test fewer symbols initially
- Use larger timeframes (1h instead of 15m)

#### Memory issues
- Process symbols one at a time
- Use smaller date ranges
- Close database connections

## 🎯 Next Steps

### Learn More
1. **Read Strategy Documentation**: See `STRATEGIES.md` for detailed strategy explanations
2. **Study Architecture**: Read `ARCHITECTURE.md` to understand the system design
3. **Explore Examples**: Look at strategy implementations in `src/trading_bot/strategies/`

### Build Your Own Strategy
1. **Start Simple**: Copy an existing strategy and modify it
2. **Add Indicators**: Use the `ta` library for technical analysis
3. **Test Thoroughly**: Backtest on multiple symbols and time periods
4. **Optimize Parameters**: Use genetic algorithms for parameter tuning

### Advanced Features
1. **Multi-Symbol Strategies**: Portfolio-based strategies
2. **Live Trading**: Extend to real-time trading (not implemented yet)
3. **Alternative Data**: Incorporate sentiment, news, on-chain data
4. **Machine Learning**: Use ML for signal generation

### Best Practices
1. **Version Control**: Commit your strategies and configurations
2. **Documentation**: Document your strategy logic and parameters
3. **Testing**: Always test on out-of-sample data
4. **Risk Management**: Never risk more than you can afford to lose

## 🎉 Congratulations!

You've successfully:
- ✅ Set up the trading bot framework
- ✅ Downloaded market data
- ✅ Run your first backtest
- ✅ Understood the results
- ✅ Learned about different strategies

You're now ready to develop and test your own trading strategies! 🚀

## 📚 Additional Resources

- **Strategy Documentation**: `STRATEGIES.md`
- **Architecture Overview**: `ARCHITECTURE.md`
- **Code Examples**: `src/trading_bot/strategies/`
- **Utility Scripts**: `scripts/`
- **Technical Analysis Library**: [TA-Lib Documentation](https://technical-analysis-library-in-python.readthedocs.io/)