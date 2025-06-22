# Trading Bot Architecture

## 🏗️ System Overview

The trading bot framework follows a modular, layered architecture designed for extensibility and maintainability.

```
┌─────────────────────────────────────────────┐
│                 User Layer                  │
├─────────────────────────────────────────────┤
│  Scripts (download_data.py, backtests)     │
├─────────────────────────────────────────────┤
│              Strategy Layer                 │
│  ┌─────────────┐ ┌─────────────┐ ┌────────┐ │
│  │EMA Crossover│ │Multi-Regime │ │  VWAP  │ │
│  │   Strategy  │ │  Strategy   │ │Strategy│ │
│  └─────────────┘ └─────────────┘ └────────┘ │
├─────────────────────────────────────────────┤
│              Engine Layer                   │
│  ┌─────────────┐ ┌─────────────┐           │
│  │  Backtest   │ │ Optimization│           │
│  │   Engine    │ │   Engine    │           │
│  └─────────────┘ └─────────────┘           │
├─────────────────────────────────────────────┤
│               Data Layer                    │
│  ┌─────────────┐ ┌─────────────┐           │
│  │Market Data  │ │   Storage   │           │
│  │  Manager    │ │  (MongoDB)  │           │
│  └─────────────┘ └─────────────┘           │
├─────────────────────────────────────────────┤
│               Core Layer                    │
│  Models, Enums, Settings, Utilities        │
└─────────────────────────────────────────────┘
```

## 🔧 Core Components

### 1. Core Layer (`src/trading_bot/core/`)

**Purpose**: Foundation types and utilities used throughout the system.

#### Key Files:
- **`models.py`**: Core data models using Pydantic
  - `MarketData`: OHLCV candle data
  - `Position`: Trading position with P&L tracking
  - `BacktestConfig`: Backtest configuration
  - `BacktestResult`: Backtest results and metrics

- **`enums.py`**: System enumerations
  - `TimeFrame`: Trading timeframes (1m, 15m, 1h, etc.)
  - `TradeStatus`: OPEN, CLOSED
  - `OrderSide`: BUY, SELL

- **`settings.py`**: Environment-based configuration using Pydantic Settings
  - MongoDB connection settings
  - Binance API configuration
  - Logging and rate limits

#### Design Principles:
- **Immutable Data**: Pydantic models ensure data consistency
- **Type Safety**: Full type hints throughout
- **Configuration**: Environment-based, no hardcoded values

### 2. Data Layer (`src/trading_bot/data/`)

**Purpose**: Market data acquisition, storage, and retrieval.

#### Components:

**`MarketDataManager`** - Main interface for data operations
```python
# High-level data operations
manager = MarketDataManager()
manager.download_and_store("BTCUSDT", TimeFrame.FIFTEEN_MINUTES, days=30)
df = manager.get_data_for_backtest("BTCUSDT", start_date, end_date)
```

**`BinanceClient`** - Binance API wrapper
```python
# Fetches raw market data from Binance
client = BinanceClient()
candles = client.fetch_historical_data("BTCUSDT", TimeFrame.FIFTEEN_MINUTES, days=30)
```

**`MongoStorage`** - MongoDB interface
```python
# Stores and retrieves candle data
storage = MongoStorage()
storage.store_candles(candles)
candles = storage.get_candles("BTCUSDT", TimeFrame.FIFTEEN_MINUTES)
```

#### Data Flow:
1. **Acquisition**: `BinanceClient` fetches from API
2. **Storage**: `MongoStorage` persists to database
3. **Retrieval**: `MarketDataManager` provides clean DataFrames
4. **Processing**: Technical indicators calculated on demand

### 3. Strategy Layer (`src/trading_bot/strategies/`)

**Purpose**: Trading logic implementation with pluggable strategies.

#### Base Strategy Interface:
```python
class BaseStrategy(ABC):
    @abstractmethod
    def buy_condition(self, row, prev_row) -> Tuple[bool, str, float]:
        """Return (should_buy, reason, stop_loss)"""
    
    @abstractmethod  
    def sell_condition(self, position, row) -> Tuple[bool, str]:
        """Return (should_sell, reason)"""
    
    @abstractmethod
    def calculate_position_size(self, balance, price, row) -> float:
        """Return position size in quote currency"""
    
    @classmethod
    @abstractmethod
    def calculate_indicators(cls, df, params) -> pd.DataFrame:
        """Add technical indicators to DataFrame"""
```

#### Strategy Implementations:

**EMA Crossover Strategy**
- **Logic**: Buy when fast EMA crosses above slow EMA
- **Indicators**: EMA fast, EMA slow
- **Simplest strategy for learning**

**Multi-Regime Strategy**  
- **Logic**: Adapts between mean reversion and momentum based on volatility
- **Indicators**: RSI, Bollinger Bands, EMA, ATR, volatility percentiles
- **Advanced strategy with regime detection**

**VWAP Statistical Strategy**
- **Logic**: Mean reversion using VWAP Z-scores
- **Indicators**: VWAP, Z-scores, volume analysis, RSI
- **Statistical approach to mean reversion**

#### Strategy Design Pattern:
1. **Initialization**: Parameters set during construction
2. **Indicator Calculation**: Class method calculates all needed indicators
3. **Signal Generation**: Instance methods generate buy/sell signals
4. **Position Sizing**: Risk-based position calculation

### 4. Engine Layer

#### Backtesting Engine (`src/trading_bot/backtesting/engine.py`)

**Purpose**: Simulate strategy performance on historical data.

**Core Algorithm**:
```python
def process_timestamp(self, row, previous_row):
    if self.current_position:
        # Check exit conditions
        self.current_position.update(row)
        sell, reason = self.strategy.sell_condition(self.current_position, row)
        if sell:
            self.sell_asset(row, reason)
    else:
        # Check entry conditions
        buy, reason, stop_loss = self.strategy.buy_condition(row, previous_row)
        if buy:
            self.buy_asset(row, reason, stop_loss)
```

**Features**:
- **Realistic Trading**: Accounts for fees, slippage
- **Position Management**: Tracks open positions, P&L
- **Multiple Symbols**: Parallel backtesting
- **Performance Metrics**: Sharpe ratio, drawdown, win rate

#### Optimization Engine (`src/trading_bot/optimization/`)

**Purpose**: Find optimal strategy parameters using genetic algorithms.

**Process**:
1. **Parameter Space Definition**: Define ranges for each parameter
2. **Population Generation**: Create random parameter combinations
3. **Fitness Evaluation**: Run backtests for each combination
4. **Evolution**: Select best performers, mutate, repeat
5. **Convergence**: Return optimal parameters

### 5. Configuration System

#### Environment-Based Settings
```python
# settings.py
class Settings(BaseSettings):
    mongodb_url: str = Field(env="MONGODB_URL")
    binance_api_key: Optional[str] = Field(env="BINANCE_API_KEY")
    
    class Config:
        env_file = ".env"
```

#### Strategy Parameters
```python
# Strategies accept parameters during initialization
strategy = EMACrossoverStrategy(
    fast_period=12,
    slow_period=26,
    stop_loss_pct=0.02,
    take_profit_pct=0.04
)
```

## 🔄 Data Flow Diagrams

### Market Data Flow
```
Binance API → BinanceClient → MarketData objects → MongoStorage → MongoDB
                                     ↓
Strategy Backtest ← DataFrame ← MarketDataManager ← MongoDB
```

### Backtesting Flow
```
Historical Data → Strategy.calculate_indicators() → DataFrame with indicators
                                     ↓
BacktestEngine.process_timestamp() → Strategy.buy_condition() → Trade Entry
                                     ↓
Position Tracking → Strategy.sell_condition() → Trade Exit → Results
```

### Optimization Flow
```
Parameter Space → Genetic Algorithm → Parameter Sets → Backtests → Fitness Scores
                        ↓
Best Parameters ← Evolution (selection, crossover, mutation) ← Population
```

## 🎯 Design Principles

### 1. Separation of Concerns
- **Data layer**: Only handles data acquisition/storage
- **Strategy layer**: Only implements trading logic  
- **Engine layer**: Only manages simulation/optimization
- **Core layer**: Only provides shared utilities

### 2. Dependency Injection
- Strategies don't know about data sources
- Engines don't know about specific strategies
- Configuration injected at runtime

### 3. Testability
- Each component can be tested in isolation
- Mock data sources for unit tests
- Clear interfaces between components

### 4. Extensibility
- New strategies implement BaseStrategy
- New data sources implement storage interface
- New optimization algorithms use same fitness interface

## 🔧 Extension Points

### Adding New Strategies
1. Inherit from `BaseStrategy`
2. Implement required methods
3. Define `INDICATOR_PARAMS` class variable
4. Add to `strategies/__init__.py`

### Adding New Data Sources
1. Implement storage interface methods
2. Create client wrapper if needed
3. Update `MarketDataManager` to use new source

### Adding New Optimization Algorithms
1. Create optimizer class in `optimization/`
2. Implement same interface as `GeneticOptimizer`
3. Use same fitness evaluation system

## 📊 Performance Considerations

### Data Management
- **MongoDB Indexing**: Compound indexes on (symbol, timeframe, timestamp)
- **Batching**: Store candles in batches, not individually
- **Caching**: Indicator calculation results cached in database

### Memory Usage
- **Streaming**: Process data in chunks for large datasets
- **Cleanup**: Close database connections after use
- **DataFrame Optimization**: Use appropriate dtypes

### Computation
- **Vectorization**: Use pandas/numpy operations over loops
- **Parallel Processing**: Multiple symbol backtests in parallel
- **Early Termination**: Stop poor-performing optimizations early

## 🔐 Security & Reliability

### API Security
- Environment variables for API keys
- Optional API usage (public data works without keys)
- Rate limiting for API calls

### Data Integrity
- Pydantic validation for all data models
- Database constraints and indexes
- Graceful error handling and logging

### Error Handling
- Comprehensive try/catch blocks
- Meaningful error messages
- Fallback mechanisms for data sources