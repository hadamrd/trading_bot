"""
SVM Sliding Window Strategy
Adaptive machine learning approach that refits every candle
"""

from typing import Any, Tuple, Optional, Set, Dict
import pandas as pd
import numpy as np
import ta
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

from ..core.models import Position
from .base import BaseStrategy


class SVMSlidingWindowStrategy(BaseStrategy):
    """
    SVM Sliding Window Strategy
    
    Key Innovation: Refits SVM model every candle using sliding window
    - Uses last N candles as training data
    - Predicts only 1 candle ahead (next direction)
    - Adapts to changing market conditions
    - Only trades when model confidence is high
    """
    
    INDICATOR_PARAMS: Set[str] = {
        'window_size', 'min_train_size', 'lookback_periods'
    }
    
    def _init_strategy(self,
                      # Sliding window parameters
                      window_size: int = 100,
                      min_train_size: int = 50,
                      lookback_periods: int = 10,
                      
                      # SVM parameters
                      svm_C: float = 1.0,
                      svm_gamma: str = 'scale',
                      confidence_threshold: float = 0.65,
                      
                      # Feature engineering
                      use_volume_features: bool = True,
                      use_volatility_features: bool = True,
                      use_momentum_features: bool = True,
                      
                      # Risk management
                      stop_loss_pct: float = 0.02,
                      take_profit_pct: float = 0.03,
                      position_size_pct: float = 0.05,
                      
                      # Performance optimization
                      retrain_frequency: int = 1,
                      cross_validate: bool = False):
        
        # Store all parameters
        self.window_size = window_size
        self.min_train_size = min_train_size
        self.lookback_periods = lookback_periods
        self.svm_C = svm_C
        self.svm_gamma = svm_gamma
        self.confidence_threshold = confidence_threshold
        self.use_volume_features = use_volume_features
        self.use_volatility_features = use_volatility_features 
        self.use_momentum_features = use_momentum_features
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position_size_pct = position_size_pct
        self.retrain_frequency = retrain_frequency
        self.cross_validate = cross_validate
        
        # State variables
        self.model = None
        self.scaler = None
        self.last_retrain_idx = -1
        self.feature_names = []

    @classmethod
    def calculate_indicators(cls, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate base indicators for feature engineering."""
        df = df.copy()
        
        if len(df) < 50:
            return df
        
        try:
            # Basic indicators
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['rsi_fast'] = ta.momentum.rsi(df['close'], window=7)
            df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=9)
            df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=21)
            df['ema_diff'] = (df['ema_fast'] - df['ema_slow']) / df['ema_slow']
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volatility
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            df['volatility'] = df['close'].rolling(window=20).std()
            
            # Volume
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price changes
            df['price_change_1'] = df['close'].pct_change(1)
            df['price_change_3'] = df['close'].pct_change(3)
            df['price_change_5'] = df['close'].pct_change(5)
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Williams %R
            df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
            
            # Target variable
            df['next_return'] = df['close'].shift(-1) / df['close'] - 1
            df['target'] = (df['next_return'] > 0).astype(int)
            
        except Exception as e:
            print(f"Error calculating SVM indicators: {e}")
        
        return df.ffill().fillna(0)

    def _create_features(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """Create feature vector for given index."""
        features = []
        
        # Basic OHLCV features
        for lookback in range(1, self.lookback_periods + 1):
            if idx - lookback >= 0:
                features.append(df.iloc[idx - lookback]['close'] / df.iloc[idx]['close'] - 1)
                features.append(df.iloc[idx - lookback]['high'] / df.iloc[idx]['close'] - 1) 
                features.append(df.iloc[idx - lookback]['low'] / df.iloc[idx]['close'] - 1)
                features.append(df.iloc[idx - lookback]['open'] / df.iloc[idx]['close'] - 1)
                
                if self.use_volume_features:
                    features.append(df.iloc[idx - lookback]['volume_ratio'])
        
        # Current technical indicators
        current_row = df.iloc[idx]
        
        if self.use_momentum_features:
            features.extend([
                current_row['rsi'] / 100,
                current_row['rsi_fast'] / 100,
                current_row['macd_histogram'],
                current_row['ema_diff'],
                current_row['price_change_1'],
                current_row['price_change_3'],
                current_row['price_change_5'],
                current_row['stoch_k'] / 100,
                current_row['williams_r'] / -100,
            ])
        
        if self.use_volatility_features:
            features.extend([
                current_row['atr'] / current_row['close'],
                current_row['volatility'] / current_row['close'],
                current_row['bb_position'],
            ])
        
        if self.use_volume_features:
            features.extend([
                current_row['volume_ratio'],
                np.log1p(current_row['volume']),
            ])
        
        return np.array(features)

    def _prepare_training_data(self, df: pd.DataFrame, end_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for the sliding window."""
        start_idx = max(0, end_idx - self.window_size)
        
        X, y = [], []
        
        for i in range(start_idx + self.lookback_periods, end_idx):
            try:
                features = self._create_features(df, i)
                target = df.iloc[i]['target']
                
                if not (np.isnan(features).any() or np.isnan(target)):
                    X.append(features)
                    y.append(target)
            except Exception:
                continue
        
        return np.array(X), np.array(y)

    def _train_model(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train SVM model on current window."""
        if len(X) < self.min_train_size or len(np.unique(y)) < 2:
            return False
        
        try:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            self.model = SVC(
                C=self.svm_C,
                gamma=self.svm_gamma,
                probability=True,
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            return True
            
        except Exception:
            return False

    def _predict(self, features: np.ndarray) -> Tuple[int, float]:
        """Make prediction with confidence."""
        if self.model is None or self.scaler is None:
            return 0, 0.0
        
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = max(probabilities)
            
            return int(prediction), float(confidence)
            
        except Exception:
            return 0, 0.0

    def buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str], float]:
        """Standard interface - not used for SVM."""
        return False, "Use run_svm_backtest method", self.stop_loss_pct

    def sell_condition(self, position: Position, row: pd.Series) -> Tuple[bool, Optional[str]]:
        """Standard risk management."""
        current_price = row['close']
        
        if current_price >= position.open_price * (1 + self.take_profit_pct):
            return True, "Take Profit"
        
        if current_price <= position.open_price * (1 - self.stop_loss_pct):
            return True, "Stop Loss"
        
        return False, None

    def calculate_position_size(self, available_balance: float, current_price: float, row: pd.Series) -> float:
        """Fixed position sizing."""
        return available_balance * self.position_size_pct

    def run_svm_backtest(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Custom backtest method with full DataFrame access."""
        print(f"🤖 Running SVM Sliding Window Backtest")
        print(f"   Data: {len(df)} candles")
        print(f"   Window size: {self.window_size}")
        
        balance = 5000.0
        initial_balance = balance
        position = None
        trades = []
        
        start_idx = self.lookback_periods + self.min_train_size
        
        for idx in range(start_idx, len(df) - 1):
            current_row = df.iloc[idx]
            
            # Retrain model
            if (idx - self.last_retrain_idx) >= self.retrain_frequency:
                X_train, y_train = self._prepare_training_data(df, idx)
                
                if self._train_model(X_train, y_train):
                    self.last_retrain_idx = idx
                    if idx % 1000 == 0:
                        print(f"   📊 Trained model at candle {idx} ({idx/len(df):.1%})")
            
            # Make prediction
            if self.model is not None:
                features = self._create_features(df, idx)
                prediction, confidence = self._predict(features)
                
                # Entry logic
                if position is None and confidence >= self.confidence_threshold:
                    if prediction == 1:
                        position_size = balance * self.position_size_pct
                        shares = position_size / current_row['close']
                        
                        position = {
                            'entry_price': current_row['close'],
                            'shares': shares,
                            'entry_idx': idx,
                            'confidence': confidence
                        }
                        balance -= position_size
                        
                        if len(trades) < 10:
                            print(f"🟢 BUY @ ${current_row['close']:.2f} | Confidence: {confidence:.1%}")
                
                # Exit logic
                elif position is not None:
                    current_price = current_row['close']
                    entry_price = position['entry_price']
                    
                    exit_reason = None
                    
                    if current_price >= entry_price * (1 + self.take_profit_pct):
                        exit_reason = "Take Profit"
                    elif current_price <= entry_price * (1 - self.stop_loss_pct):
                        exit_reason = "Stop Loss"
                    elif prediction == 0 and confidence >= self.confidence_threshold:
                        exit_reason = "Model Signal"
                    
                    if exit_reason:
                        exit_value = position['shares'] * current_price
                        profit = exit_value - (position['shares'] * entry_price)
                        profit_pct = (current_price / entry_price - 1) * 100
                        
                        balance += exit_value
                        
                        trade = {
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'profit': profit,
                            'profit_pct': profit_pct,
                            'confidence': position['confidence'],
                            'exit_reason': exit_reason,
                            'duration': idx - position['entry_idx']
                        }
                        trades.append(trade)
                        
                        if len(trades) <= 10:
                            print(f"🔴 SELL @ ${current_price:.2f} | P&L: ${profit:.2f} ({profit_pct:+.2f}%) | {exit_reason}")
                        
                        position = None
        
        # Close final position
        if position is not None:
            final_price = df.iloc[-1]['close']
            exit_value = position['shares'] * final_price
            profit = exit_value - (position['shares'] * position['entry_price'])
            balance += exit_value
            
            trades.append({
                'entry_price': position['entry_price'],
                'exit_price': final_price,
                'profit': profit,
                'profit_pct': (final_price / position['entry_price'] - 1) * 100,
                'confidence': position['confidence'],
                'exit_reason': 'End of Data',
                'duration': len(df) - position['entry_idx']
            })
        
        # Calculate results
        total_return = balance - initial_balance
        total_return_pct = (total_return / initial_balance) * 100
        
        if trades:
            winning_trades = [t for t in trades if t['profit'] > 0]
            win_rate = len(winning_trades) / len(trades)
            avg_profit = np.mean([t['profit'] for t in trades])
            avg_confidence = np.mean([t['confidence'] for t in trades])
        else:
            win_rate = 0
            avg_profit = 0
            avg_confidence = 0
        
        return {
            'total_trades': len(trades),
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'win_rate': win_rate,
            'avg_profit_per_trade': avg_profit,
            'avg_confidence': avg_confidence,
            'final_balance': balance,
            'trades': trades
        }

    def __str__(self) -> str:
        return f"SVMSlidingWindow(window={self.window_size}, confidence={self.confidence_threshold:.1%})"