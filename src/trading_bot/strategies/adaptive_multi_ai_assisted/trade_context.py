"""
Trade context capture and analysis
Extracts comprehensive market context for AI validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

from .trade_validation import ContextSummary, ValidationRequest, CandleData


@dataclass
class TechnicalLevels:
    """Key technical levels for context"""
    support: float
    resistance: float
    recent_high: float
    recent_low: float
    pivot_point: float


class TradeContextAnalyzer:
    """Analyzes market data to extract trading context"""
    
    def __init__(self):
        self.lookback_periods = {
            'short': 20,    # ~1.5 hours on 5min
            'medium': 50,   # ~4 hours on 5min  
            'long': 200     # ~16 hours on 5min
        }
    
    def extract_context(self, 
                       row: pd.Series, 
                       df: pd.DataFrame, 
                       current_idx: int,
                       symbol: str,
                       timeframe: str) -> ContextSummary:
        """Extract comprehensive market context from data"""
        
        # Get recent data windows
        short_window = self._get_window(df, current_idx, self.lookback_periods['short'])
        medium_window = self._get_window(df, current_idx, self.lookback_periods['medium'])
        long_window = self._get_window(df, current_idx, self.lookback_periods['long'])
        
        # Calculate price changes
        price_changes = self._calculate_price_changes(row, short_window, medium_window)
        
        # Extract technical levels
        tech_levels = self._calculate_technical_levels(short_window, medium_window)
        
        # Get market timing context
        timing_info = self._get_timing_context(row.name if hasattr(row, 'name') else datetime.now())
        
        return ContextSummary(
            current_price=float(row['close']),
            price_change_1h=price_changes['1h'],
            price_change_24h=price_changes['24h'] if price_changes['24h'] is not None else 0.0,
            
            rsi_level=float(row.get('rsi', 50)),
            volume_vs_average=float(row.get('volume_ratio', 1.0)),
            volatility_percentile=float(row.get('volatility_percentile', 0.5)),
            
            detected_regime=str(row.get('regime', 'unknown')),
            volatility_mood=str(row.get('mood', 'normal')),
            
            key_technical_levels={
                'support': tech_levels.support,
                'resistance': tech_levels.resistance,
                'recent_high': tech_levels.recent_high,
                'recent_low': tech_levels.recent_low,
                'pivot': tech_levels.pivot_point
            },
            
            market_hours=timing_info['session'],
            day_of_week=timing_info['day_name']
        )
    
    def _get_window(self, df: pd.DataFrame, current_idx: int, periods: int) -> pd.DataFrame:
        """Get data window with bounds checking"""
        start_idx = max(0, current_idx - periods)
        end_idx = min(len(df), current_idx + 1)
        return df.iloc[start_idx:end_idx]
    
    def _calculate_price_changes(self, 
                                row: pd.Series, 
                                short_window: pd.DataFrame, 
                                medium_window: pd.DataFrame) -> Dict[str, Optional[float]]:
        """Calculate various price change metrics"""
        
        current_price = row['close']
        
        # 1 hour change (12 periods on 5min)
        hour_change = None
        if len(short_window) >= 12:
            hour_ago_price = short_window.iloc[-12]['close']
            hour_change = (current_price - hour_ago_price) / hour_ago_price
        
        # 24 hour change (288 periods on 5min, use what we have)
        day_change = None
        if len(medium_window) >= 50:  # At least some longer-term data
            day_ago_price = medium_window.iloc[0]['close']
            day_change = (current_price - day_ago_price) / day_ago_price
        
        return {
            '1h': hour_change if hour_change is not None else 0.0,
            '24h': day_change
        }
    
    def _calculate_technical_levels(self, 
                                   short_window: pd.DataFrame, 
                                   medium_window: pd.DataFrame) -> TechnicalLevels:
        """Calculate key technical support/resistance levels"""
        
        if len(short_window) < 5:
            # Fallback for insufficient data
            current_price = short_window['close'].iloc[-1] if len(short_window) > 0 else 0
            return TechnicalLevels(
                support=current_price * 0.99,
                resistance=current_price * 1.01,
                recent_high=current_price,
                recent_low=current_price,
                pivot_point=current_price
            )
        
        # Recent high/low (short window)
        recent_high = short_window['high'].max()
        recent_low = short_window['low'].min()
        
        # Support/resistance (medium window if available)
        window_for_levels = medium_window if len(medium_window) >= 20 else short_window
        
        # Simple support/resistance using recent ranges
        price_range = window_for_levels['close']
        resistance = price_range.quantile(0.9)  # 90th percentile as resistance
        support = price_range.quantile(0.1)     # 10th percentile as support
        
        # Pivot point (simple calculation)
        if len(window_for_levels) > 0:
            high = window_for_levels['high'].iloc[-1]
            low = window_for_levels['low'].iloc[-1]
            close = window_for_levels['close'].iloc[-1]
            pivot = (high + low + close) / 3
        else:
            pivot = short_window['close'].iloc[-1]
        
        return TechnicalLevels(
            support=float(support),
            resistance=float(resistance),
            recent_high=float(recent_high),
            recent_low=float(recent_low),
            pivot_point=float(pivot)
        )
    
    def _get_timing_context(self, timestamp: datetime) -> Dict[str, str]:
        """Determine market session and timing context"""
        
        hour_utc = timestamp.hour
        day_name = timestamp.strftime('%A')
        
        # Determine market session (simplified)
        if 0 <= hour_utc < 8:
            session = "asian"
        elif 8 <= hour_utc < 16:
            session = "european"
        elif 16 <= hour_utc < 24:
            session = "american"
        else:
            session = "off_hours"
        
        return {
            'session': session,
            'day_name': day_name,
            'hour_utc': hour_utc
        }


class TradeHistoryAnalyzer:
    """Analyzes recent trade history for context"""
    
    def __init__(self):
        self.recent_trades = []
    
    def add_trade_result(self, 
                        signal_reason: str,
                        entry_price: float,
                        exit_price: Optional[float],
                        duration_minutes: Optional[int],
                        outcome: str):
        """Add completed trade to history"""
        
        trade_record = {
            'timestamp': datetime.now(),
            'signal_reason': signal_reason,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'duration_minutes': duration_minutes,
            'outcome': outcome,
            'return_pct': ((exit_price / entry_price) - 1) * 100 if exit_price else None
        }
        
        self.recent_trades.append(trade_record)
        
        # Keep only recent trades (last 50)
        if len(self.recent_trades) > 50:
            self.recent_trades = self.recent_trades[-50:]
    
    def get_similar_trades(self, signal_reason: str, lookback_days: int = 7) -> List[Dict]:
        """Get trades with similar signals from recent history"""
        
        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        
        similar_trades = []
        for trade in self.recent_trades:
            if trade['timestamp'] >= cutoff_time:
                # Check for similar signal patterns
                if self._signals_similar(signal_reason, trade['signal_reason']):
                    similar_trades.append({
                        'signal': trade['signal_reason'],
                        'outcome': trade['outcome'],
                        'return_pct': trade['return_pct'],
                        'duration_min': trade['duration_minutes']
                    })
        
        return similar_trades[-5:]  # Return last 5 similar trades
    
    def _signals_similar(self, signal1: str, signal2: str) -> bool:
        """Check if two signals are similar"""
        
        # Extract key components
        signal1_parts = signal1.lower().split()
        signal2_parts = signal2.lower().split()
        
        # Check for common keywords
        common_keywords = set(signal1_parts) & set(signal2_parts)
        
        # Similar if they share significant keywords
        return len(common_keywords) >= 2 or any(
            keyword in signal1.lower() and keyword in signal2.lower()
            for keyword in ['oversold', 'cross', 'breakout', 'bounce', 'reversal']
        )
    
    def get_recent_performance_stats(self, lookback_days: int = 7) -> Dict[str, float]:
        """Get recent trading performance statistics"""
        
        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        recent = [t for t in self.recent_trades if t['timestamp'] >= cutoff_time]
        
        if not recent:
            return {'win_rate': 0.0, 'avg_return': 0.0, 'total_trades': 0}
        
        wins = len([t for t in recent if t['outcome'] == 'WIN'])
        total = len(recent)
        returns = [t['return_pct'] for t in recent if t['return_pct'] is not None]
        
        return {
            'win_rate': wins / total if total > 0 else 0.0,
            'avg_return': np.mean(returns) if returns else 0.0,
            'total_trades': total
        }


class ValidationRequestBuilder:
    """Builds complete validation requests with all context"""
    
    def __init__(self):
        self.context_analyzer = TradeContextAnalyzer()
        self.history_analyzer = TradeHistoryAnalyzer()
    
    def build_request(self,
                     symbol: str,
                     timeframe: str,
                     signal_reason: str,
                     entry_price: float,
                     stop_loss: float,
                     position_size: float,
                     row: pd.Series,
                     df: pd.DataFrame,
                     current_idx: int,
                     strategy_confidence: float = 1.0,
                     portfolio_exposure: float = 0.0,
                     candles_lookback: int = 15) -> ValidationRequest:
        """Build complete validation request with all context including recent candles"""
        
        # Extract market context
        context = self.context_analyzer.extract_context(
            row, df, current_idx, symbol, timeframe
        )
        
        # Get similar trade history
        similar_trades = self.history_analyzer.get_similar_trades(signal_reason)
        
        # Extract recent candles for price action history
        recent_candles = self._extract_recent_candles(df, current_idx, candles_lookback)
        
        return ValidationRequest(
            symbol=symbol,
            timeframe=timeframe,
            signal_reason=signal_reason,
            proposed_entry_price=entry_price,
            proposed_stop_loss=stop_loss,
            proposed_position_size=position_size,
            context=context,
            recent_candles=recent_candles,
            recent_similar_trades=similar_trades,
            strategy_confidence=strategy_confidence,
            current_portfolio_exposure=portfolio_exposure
        )
    
    def add_trade_outcome(self, *args, **kwargs):
        """Delegate to history analyzer"""
        self.history_analyzer.add_trade_result(*args, **kwargs)
    
    def get_performance_stats(self, lookback_days: int = 7) -> Dict[str, float]:
        """Get recent performance statistics"""
        return self.history_analyzer.get_recent_performance_stats(lookback_days)
    
    def _extract_recent_candles(self, df: pd.DataFrame, current_idx: int, lookback: int) -> List[CandleData]:
        """Extract recent candles for price action history"""
        
        start_idx = max(0, current_idx - lookback)
        end_idx = current_idx + 1
        
        recent_df = df.iloc[start_idx:end_idx]
        candles = []
        
        for i, (timestamp, row) in enumerate(recent_df.iterrows()):
            # Calculate percentage change from previous candle
            change_pct = None
            if i > 0:
                prev_close = recent_df.iloc[i-1]['close']
                change_pct = ((row['close'] - prev_close) / prev_close) * 100
            
            candle = CandleData(
                timestamp=timestamp.strftime('%m-%d %H:%M') if hasattr(timestamp, 'strftime') else str(timestamp),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']),
                volume_ratio=float(row.get('volume_ratio', 1.0)) if 'volume_ratio' in row else None,
                rsi=float(row.get('rsi', 50)) if 'rsi' in row and pd.notna(row['rsi']) else None,
                change_pct=change_pct
            )
            candles.append(candle)
        
        return candles
