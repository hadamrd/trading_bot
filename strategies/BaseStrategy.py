from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Optional, Dict, Any, ClassVar, Set
from tradingbot2.TradePosition import TradePosition

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies with separated indicator logic
    """
    
    # Class variable defining which parameters are used for indicators
    INDICATOR_PARAMS: ClassVar[Set[str]] = set()
    
    @classmethod
    def get_indicator_params(cls, strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract only the parameters needed for indicator calculation
        
        Args:
            strategy_params: Dictionary containing all strategy parameters
            
        Returns:
            Dict containing only the parameters needed for indicators
        """
        return {k: v for k, v in strategy_params.items() if k in cls.INDICATOR_PARAMS}
    
    @classmethod
    @abstractmethod
    def calculate_indicators(cls, df: pd.DataFrame, indicator_params: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate strategy-specific indicators
        
        Args:
            df: DataFrame containing OHLCV data
            indicator_params: Dictionary containing only the parameters needed for indicators
            
        Returns:
            DataFrame with added indicator columns
        """
        pass
    
    @abstractmethod
    def _init_strategy(self, **params):
        """
        Initialize strategy-specific parameters
        
        Args:
            **params: Strategy parameters as keyword arguments
        """
        pass
    
    def __init__(self, **strategy_params):
        """
        Initialize the strategy with all parameters
        
        Args:
            **strategy_params: All strategy parameters as keyword arguments
        """
        self._init_strategy(**strategy_params)
        # Store the full set of parameters for reference
        self._strategy_params = strategy_params
    
    @abstractmethod
    def buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str], float]:
        """
        Determine if a buy signal is present
        
        Args:
            row: Current DataFrame row
            prev_row: Previous DataFrame row
            
        Returns:
            Tuple of (should_buy: bool, reason: Optional[str], stop_loss: float)
        """
        pass
    
    @abstractmethod
    def sell_condition(self, position: TradePosition, row: pd.Series) -> Tuple[bool, Optional[str]]:
        """
        Determine if a sell signal is present
        
        Args:
            position: Current trade position
            row: Current DataFrame row
            
        Returns:
            Tuple of (should_sell: bool, reason: Optional[str])
        """
        pass
    
    @abstractmethod
    def calculate_take_profit(self, position: TradePosition, row: pd.Series) -> float:
        """
        Calculate take profit level for a position
        
        Args:
            position: Current trade position
            row: Current DataFrame row
            
        Returns:
            Take profit price level
        """
        pass
    
    @abstractmethod
    def calculate_stop_loss(self, position: TradePosition, row: pd.Series) -> float:
        """
        Calculate stop loss level for a position
        
        Args:
            position: Current trade position
            row: Current DataFrame row
            
        Returns:
            Stop loss price level
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, available_balance: float, current_price: float, row: pd.Series) -> float:
        """
        Calculate the position size for a new trade
        
        Args:
            available_balance: Available balance for trading
            current_price: Current asset price
            row: Current DataFrame row
            
        Returns:
            Position size in quote currency
        """
        pass
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """
        Get all strategy parameters
        
        Returns:
            Dictionary containing all strategy parameters
        """
        return self._strategy_params.copy()
    
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters
        
        Returns:
            True if parameters are valid, False otherwise
        
        This method can be overridden by concrete strategies to add parameter validation.
        """
        return True
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for strategy use, including calculating indicators
        
        Args:
            df: Raw DataFrame with OHLCV data
            
        Returns:
            Prepared DataFrame with indicators
        """
        indicator_params = self.get_indicator_params(self._strategy_params)
        return self.calculate_indicators(df, indicator_params)
    
    @classmethod
    def get_required_columns(cls) -> Set[str]:
        """
        Get the required columns for this strategy
        
        Returns:
            Set of column names that must be present in the DataFrame
        """
        return {'open', 'high', 'low', 'close', 'volume', 'timestamp'}
    
    @classmethod
    def verify_dataframe(cls, df: pd.DataFrame) -> bool:
        """
        Verify that a DataFrame has all required columns
        
        Args:
            df: DataFrame to verify
            
        Returns:
            True if DataFrame has all required columns, False otherwise
        """
        required_columns = cls.get_required_columns()
        return all(col in df.columns for col in required_columns)
