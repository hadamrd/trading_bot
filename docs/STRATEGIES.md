# Trading Strategies Documentation

## 📊 Strategy Overview

This framework includes three main trading strategies, each implementing different market approaches:

| Strategy | Type | Best For | Complexity |
|----------|------|----------|------------|
| **EMA Crossover** | Trend Following | Trending markets | Simple |
| **Multi-Regime** | Adaptive | Volatile/changing markets | Advanced |
| **VWAP Statistical** | Mean Reversion | Range-bound markets | Intermediate |

## 🎯 Strategy Interface

All strategies inherit from `BaseStrategy` and must implement:

```python
class BaseStrategy(ABC):
    # Class variable: parameters needed for indicators
    INDICATOR_PARAMS: Set[str] = {'param1', 'param2'}
    
    @classmethod
    def calculate_indicators(cls, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Calculate technical indicators"""
    
    def buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, str, float]:
        """Return (should_buy, reason, stop_loss_percentage)"""
    
    def sell_condition(self, position: Position, row: pd.Series) -> Tuple[bool, str]:
        """Return (should_sell, reason)"""
        
    def calculate_position_size(self, balance: float, price: float, row: pd.Series) -> float:
        """Return position size in quote currency"""
```

---

## 1. 📈 EMA Crossover Strategy

**File**: `src/trading_bot/strategies/ema_crossover.py`

### Strategy Logic
- **Buy Signal**: Fast EMA crosses above slow EMA (bullish crossover)
- **Sell Signal**: Fast EMA crosses below slow EMA, or stop-loss/take-profit hit
- **Type**: Trend following
- **Best Markets**: Trending markets with clear directional moves

### Parameters

```python
EMACrossoverStrategy(
    fast_period=12,           # Fast EMA period
    slow_period=26,           # Slow EMA period  
    stop_loss_pct=0.02,       # Stop loss percentage (2%)
    take_profit_pct=0.04,     # Take profit percentage (4%)
    position_size_pct=0.1     # Position size (10% of balance)
)
```

### Technical Indicators

| Indicator | Purpose | Calculation |
|-----------|---------|-------------|
| `ema_fast` | Fast trend line | EMA(close, fast_period) |
| `ema_slow` | Slow trend line | EMA(close, slow_period) |
| `ema_diff` | Trend strength | ema_fast - ema_slow |
| `bullish_cross` | Buy signal | ema_diff > 0 AND prev_ema_diff <= 0 |
| `bearish_cross` | Sell signal | ema_diff < 0 AND prev_ema_diff >= 0 |

### Entry Conditions
```python
def buy_condition(self, row, prev_row):
    # Need both EMAs calculated
    if pd.isna(row['ema_fast']) or pd.isna(row['ema_slow']):
        return False, None, 0.0
    
    # Bullish crossover
    if row['bullish_cross']:
        return True, "EMA bullish crossover", self.stop_loss_pct
```

### Exit Conditions
1. **Take Profit**: Price reaches take_profit_pct above entry
2. **Stop Loss**: Price reaches stop_loss_pct below entry  
3. **Signal Exit**: Bearish crossover occurs

### Usage Example
```python
from trading_bot.strategies.ema_crossover import EMACrossoverStrategy
from trading_bot.backtesting.engine import BacktestEngine

# Create strategy
strategy = EMACrossoverStrategy(
    fast_period=9,           # Faster for crypto
    slow_period=21,          # Standard
    stop_loss_pct=0.025,     # 2.5% stop loss
    take_profit_pct=0.05,    # 5% take profit
    position_size_pct=0.08   # 8% position size
)

# Run backtest
engine = BacktestEngine(config, strategy)
results = engine.run()
```

### Optimization Parameters
- **fast_period**: 5-20 (shorter for crypto volatility)
- **slow_period**: 15-50 (longer trend confirmation)
- **stop_loss_pct**: 0.01-0.05 (1%-5%)
- **take_profit_pct**: 0.02-0.08 (2%-8%)

---

## 2. 🌊 Multi-Regime Strategy

**File**: `src/trading_bot/strategies/multi_regime.py`

### Strategy Logic
- **Regime Detection**: Uses volatility percentiles to identify market conditions
- **Low Volatility**: Mean reversion approach (buy oversold, sell overbought)
- **High Volatility**: Momentum approach (buy breakouts, trend following)
- **Medium Volatility**: Mixed signals with weaker thresholds

### Parameters

```python
MultiRegimeStrategy(
    # Regime detection
    volatility_threshold_high=0.7,    # High vol if >70th percentile
    volatility_threshold_low=0.3,     # Low vol if <30th percentile
    volatility_lookback=50,           # Periods for volatility ranking
    
    # Mean reversion (low vol)
    rsi_oversold=30,                  # RSI oversold level
    rsi_overbought=70,                # RSI overbought level
    bb_period=20,                     # Bollinger Band period
    bb_std=2.0,                       # Bollinger Band standard deviations
    
    # Momentum (high vol)
    ema_fast=9,                       # Fast EMA
    ema_slow=21,                      # Slow EMA
    
    # Risk management
    stop_loss_atr=2.0,                # Stop loss in ATR units
    take_profit_atr=3.0,              # Take profit in ATR units
    position_size_pct=0.02            # Position size percentage
)
```

### Technical Indicators

| Indicator | Purpose | Calculation |
|-----------|---------|-------------|
| `volatility_percentile` | Regime detection | ATR rank over lookback period |
| `volatility_regime` | Current regime | 'low_vol', 'medium_vol', 'high_vol' |
| `rsi` | Mean reversion | RSI(close, 14) |
| `bb_upper/lower` | Mean reversion | Bollinger Bands |
| `bb_position` | Price within bands | (close - bb_lower) / (bb_upper - bb_lower) |
| `ema_fast/slow` | Momentum | EMA indicators |
| `ema_diff` | Trend strength | ema_fast - ema_slow |
| `atr` | Volatility | Average True Range |

### Entry Logic by Regime

#### Low Volatility (Mean Reversion)
```python
oversold_condition = (
    row['rsi'] < self.rsi_oversold and
    row['bb_position'] < 0.2 and           # Near lower BB
    row['volume_ratio'] > 1.2              # Volume confirmation
)
```

#### High Volatility (Momentum)
```python
momentum_condition = (
    row['ema_diff'] > 0 and                # Uptrend
    row['ema_diff'] > prev_row['ema_diff'] and  # Accelerating
    row['rsi'] > 45 and                    # Not oversold
    row['volume_ratio'] > 1.5              # Strong volume
)
```

#### Medium Volatility (Mixed)
- Weaker versions of both conditions
- Golden cross signals (EMA crossover)
- Conservative mean reversion

### Exit Conditions
1. **ATR-based stops**: Dynamic based on current volatility
2. **Strategy-specific exits**:
   - Mean reversion: RSI overbought, upper Bollinger Band
   - Momentum: EMA bearish crossover
3. **Risk-reward targets**: Based on ATR multiples

### Usage Example
```python
strategy = MultiRegimeStrategy(
    volatility_threshold_high=0.75,  # More conservative regime detection
    rsi_oversold=25,                 # Deeper oversold for crypto
    ema_fast=12,                     # Slightly slower EMAs
    ema_slow=26,
    stop_loss_atr=1.5,              # Tighter stops
    take_profit_atr=2.5,            # Lower targets
    position_size_pct=0.03          # Larger positions for fewer signals
)
```

### Regime Characteristics
- **Low Vol (0-30th percentile)**: Range-bound, mean-reverting
- **Medium Vol (30-70th percentile)**: Mixed conditions
- **High Vol (70-100th percentile)**: Trending, momentum-driven

---

## 3. 📊 VWAP Statistical Strategy

**File**: `src/trading_bot/strategies/vwap_statistical.py`

### Strategy Logic
- **VWAP Z-Score**: Measures how far price deviates from VWAP statistically
- **Entry**: Buy when Z-score shows significant negative deviation (oversold)
- **Exit**: Sell when Z-score returns to positive territory (reversion)
- **Confirmations**: Volume, RSI, session timing, trend filters

### Parameters

```python
VWAPStatisticalStrategy(
    # VWAP calculation
    vwap_period=20,                   # VWAP calculation period
    zscore_period=20,                 # Z-score lookback period
    
    # Entry/exit thresholds
    zscore_entry_threshold=-1.5,      # Buy when Z-score < -1.5
    zscore_exit_threshold=0.5,        # Sell when Z-score > 0.5
    
    # Volume confirmation
    min_volume_ratio=1.2,             # Minimum volume spike
    
    # RSI filters
    rsi_oversold=35,                  # RSI confirmation level
    rsi_overbought=65,                # RSI exit level
    
    # Risk management
    stop_loss_atr=2.5,                # ATR-based stop loss
    take_profit_multiple=2.0,         # Risk:reward ratio
    
    # Additional filters
    require_active_session=True,      # Trade only during active hours
    max_vwap_slope=0.002,            # Avoid strong trends
    position_size_pct=0.03           # Position size
)
```

### Technical Indicators

| Indicator | Purpose | Calculation |
|-----------|---------|-------------|
| `vwap` | Volume-weighted price | VWAP(high, low, close, volume, period) |
| `vwap_deviation_pct` | Price vs VWAP | (close - vwap) / vwap |
| `vwap_zscore` | Statistical deviation | (deviation - mean) / std |
| `vwap_slope` | VWAP trend | vwap.diff() / vwap.shift() |
| `volume_ratio` | Volume spike | volume / volume_sma |
| `rsi` | Momentum | RSI(close, 14) |
| `is_active_session` | Time filter | Active trading hours |

### Entry Conditions
```python
def buy_condition(self, row, prev_row):
    # Primary signal: Statistical oversold
    zscore_signal = row['vwap_zscore'] <= self.zscore_entry_threshold
    
    # Volume confirmation
    volume_confirmation = row['volume_ratio'] >= self.min_volume_ratio
    
    # RSI confirmation
    rsi_confirmation = row['rsi'] <= self.rsi_oversold
    
    # Trend filter: avoid strong trends
    trend_filter = abs(row['vwap_slope']) <= self.max_vwap_slope
    
    # Session filter
    session_filter = row['is_active_session'] or not self.require_active_session
    
    return all([zscore_signal, volume_confirmation, rsi_confirmation, 
                trend_filter, session_filter])
```

### Exit Conditions
1. **Z-Score Reversion**: Z-score > exit_threshold
2. **Risk-Reward Target**: Price reaches calculated target
3. **RSI Overbought**: RSI > overbought_level
4. **ATR Stop Loss**: Price hits volatility-based stop
5. **Failed Reversion**: Z-score > 2.0 (emergency exit)

### Statistical Features

#### Z-Score Calculation
```python
# Rolling statistics
rolling_mean = vwap_deviation_pct.rolling(zscore_period).mean()
rolling_std = vwap_deviation_pct.rolling(zscore_period).std()
zscore = (vwap_deviation_pct - rolling_mean) / rolling_std
```

#### Position Sizing by Signal Strength
```python
def calculate_position_size(self, balance, price, row):
    base_size = balance * self.position_size_pct
    
    # Stronger signals get larger positions
    zscore = abs(row['vwap_zscore'])
    if zscore >= 2.0:
        multiplier = 1.5
    elif zscore >= 1.5:
        multiplier = 1.2
    else:
        multiplier = 1.0
    
    return base_size * multiplier
```

### Usage Example
```python
strategy = VWAPStatisticalStrategy(
    vwap_period=14,                  # Shorter VWAP for crypto
    zscore_entry_threshold=-2.0,     # Stronger signal required
    zscore_exit_threshold=0.3,       # Earlier exit
    min_volume_ratio=1.5,            # Higher volume requirement
    rsi_oversold=30,                 # Standard RSI level
    stop_loss_atr=2.0,              # Tighter stop
    take_profit_multiple=1.5,        # Conservative target
    require_active_session=False,    # Trade 24/7 for crypto
    position_size_pct=0.04          # Larger positions
)
```

### Session Filters (if enabled)
- **European Morning**: 8:00-11:00 UTC
- **US Morning**: 14:00-17:00 UTC  
- **Asian Session**: 20:00-23:00 UTC

---

## 🎯 Strategy Selection Guide

### Choose EMA Crossover When:
- ✅ Learning the framework
- ✅ Trending markets
- ✅ Simple, interpretable signals
- ✅ Lower-frequency trading preferred

### Choose Multi-Regime When:
- ✅ Volatile, changing market conditions
- ✅ Want adaptive strategy
- ✅ Comfortable with complexity
- ✅ Good for multiple market types

### Choose VWAP Statistical When:
- ✅ Range-bound markets
- ✅ Want statistical rigor
- ✅ Intraday mean reversion
- ✅ Volume-based confirmation important

## 🔧 Creating Custom Strategies

### 1. Basic Template
```python
from .base import BaseStrategy
from typing import Tuple, Optional, Set, Dict, Any
import pandas as pd

class MyStrategy(BaseStrategy):
    # Define parameters used for indicators
    INDICATOR_PARAMS: Set[str] = {'my_param1', 'my_param2'}
    
    @classmethod
    def calculate_indicators(cls, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        # Add your indicators here
        df['my_indicator'] = some_calculation(df['close'])
        return df
    
    def _init_strategy(self, my_param1=10, my_param2=0.02):
        self.my_param1 = my_param1
        self.my_param2 = my_param2
    
    def buy_condition(self, row, prev_row) -> Tuple[bool, Optional[str], float]:
        if row['my_indicator'] > some_threshold:
            return True, "My condition met", 0.02
        return False, None, 0.0
    
    def sell_condition(self, position, row) -> Tuple[bool, Optional[str]]:
        # Add exit logic
        return False, None
    
    def calculate_position_size(self, balance, price, row) -> float:
        return balance * 0.1  # 10% of balance
```

### 2. Register Strategy
Add to `src/trading_bot/strategies/__init__.py`:
```python
from .my_strategy import MyStrategy

__all__ = [
    "BaseStrategy",
    "EMACrossoverStrategy", 
    "MultiRegimeStrategy",
    "VWAPStatisticalStrategy",
    "MyStrategy"  # Add your strategy
]
```

### 3. Test Strategy
```python
# Test your strategy
strategy = MyStrategy(my_param1=15, my_param2=0.03)
engine = BacktestEngine(config, strategy)
results = engine.run()
```

## 📊 Performance Tuning

### Parameter Optimization
1. **Start Simple**: Begin with default parameters
2. **One at a Time**: Optimize one parameter at a time initially
3. **Use Genetic Algorithm**: For multi-parameter optimization
4. **Cross-Validation**: Test on different time periods
5. **Walk-Forward**: Use rolling optimization windows

### Common Optimizations
- **Entry Thresholds**: Tighter thresholds = fewer, higher-quality trades
- **Exit Rules**: Earlier exits = higher win rate, lower average profit
- **Position Sizing**: Larger positions = higher returns, higher risk
- **Risk Management**: Tighter stops = lower drawdown, more stopped out

### Avoiding Overfitting
- **Out-of-Sample Testing**: Reserve data for final validation
- **Multiple Markets**: Test on different symbols
- **Multiple Timeframes**: Validate on different timeframes
- **Walk-Forward Analysis**: Use rolling optimization windows
- **Parameter Stability**: Prefer strategies that work with wide parameter ranges