"""
Base strategy interface for all trading strategies.
Restored to match the original working BaseStrategy.py interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Set, ClassVar, Tuple
import pandas as pd
from decimal import Decimal

from ..core.models import Position


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Restored to match original working interface.
    """
    
    # Class variable defining which parameters are used for indicators
    INDICATOR_PARAMS: ClassVar[Set[str]] = set()
    
    def __init__(self, **parameters):
        """Initialize the strategy with given parameters."""
        self.parameters = parameters
        self.name = self.__class__.__name__
        self._init_strategy(**parameters)
    
    @abstractmethod
    def _init_strategy(self, **params):
        """Initialize strategy-specific parameters."""
        pass
    
    @classmethod
    def get_indicator_params(cls, all_params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters needed for indicator calculation."""
        return {k: v for k, v in all_params.items() if k in cls.INDICATOR_PARAMS}
    
    @classmethod
    @abstractmethod
    def calculate_indicators(cls, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate technical indicators for the strategy."""
        pass
    
    @abstractmethod
    def buy_condition(self, row: pd.Series, prev_row: pd.Series) -> Tuple[bool, Optional[str], float]:
        """
        Determine if a buy signal is present.
        
        Args:
            row: Current DataFrame row
            prev_row: Previous DataFrame row
            
        Returns:
            Tuple of (should_buy: bool, reason: Optional[str], stop_loss: float)
        """
        pass
    
    @abstractmethod
    def sell_condition(self, position: Position, row: pd.Series) -> Tuple[bool, Optional[str]]:
        """
        Determine if a sell signal is present.
        
        Args:
            position: Current trade position
            row: Current DataFrame row
            
        Returns:
            Tuple of (should_sell: bool, reason: Optional[str])
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, available_balance: float, current_price: float, row: pd.Series) -> float:
        """
        Calculate the position size for a new trade.
        
        Args:
            available_balance: Available balance for trading
            current_price: Current asset price
            row: Current DataFrame row
            
        Returns:
            Position size in quote currency
        """
        pass
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data by calculating all required indicators."""
        indicator_params = self.get_indicator_params(self.parameters)
        return self.calculate_indicators(df, indicator_params)
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get all strategy parameters."""
        return self.parameters.copy()
    
    def __str__(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.parameters.items())})"