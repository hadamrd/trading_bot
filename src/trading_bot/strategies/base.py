"""
Base strategy interface for all trading strategies.
Fixed to match the original working interface exactly.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd

from ..core.models import Position


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Restored to match original working interface exactly.
    """

    # Class variable defining which parameters are used for indicators
    INDICATOR_PARAMS: ClassVar[set[str]] = set()

    @classmethod
    def get_indicator_params(cls, strategy_params: dict[str, Any]) -> dict[str, Any]:
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
    def calculate_indicators(cls, df: pd.DataFrame, indicator_params: dict[str, Any]) -> pd.DataFrame:
        """
        Calculate strategy-specific indicators

        Args:
            df: DataFrame containing OHLCV data
            indicator_params: Dictionary containing only the parameters needed for indicators

        Returns:
            DataFrame with added indicator columns
        """

    @abstractmethod
    def _init_strategy(self, **params):
        """
        Initialize strategy-specific parameters

        Args:
            **params: Strategy parameters as keyword arguments
        """

    def __init__(self, **strategy_params):
        """
        Initialize the strategy with all parameters

        Args:
            **strategy_params: All strategy parameters as keyword arguments
        """
        self._init_strategy(**strategy_params)
        # Store the full set of parameters for reference
        self._strategy_params = strategy_params
        self.name = self.__class__.__name__

    @abstractmethod
    def buy_condition(self, row: pd.Series, prev_row: pd.Series) -> tuple[bool, str | None, float]:
        """
        Determine if a buy signal is present

        Args:
            row: Current DataFrame row
            prev_row: Previous DataFrame row

        Returns:
            Tuple of (should_buy: bool, reason: Optional[str], stop_loss: float)
        """

    @abstractmethod
    def sell_condition(self, position: Position, row: pd.Series) -> tuple[bool, str | None]:
        """
        Determine if a sell signal is present

        Args:
            position: Current trade position
            row: Current DataFrame row

        Returns:
            Tuple of (should_sell: bool, reason: Optional[str])
        """

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

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data by calculating all required indicators.

        Args:
            df: Raw DataFrame with OHLCV data

        Returns:
            DataFrame with indicators calculated
        """
        indicator_params = self.get_indicator_params(self._strategy_params)
        return self.calculate_indicators(df, indicator_params)

    def get_strategy_params(self) -> dict[str, Any]:
        """Get all strategy parameters."""
        return self._strategy_params.copy()

    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters

        Returns:
            True if parameters are valid, False otherwise

        This method can be overridden by concrete strategies to add parameter validation.
        """
        return True

    @classmethod
    def get_required_columns(cls) -> set[str]:
        """
        Get the required columns for this strategy

        Returns:
            Set of column names that must be present in the DataFrame
        """
        return {'open', 'high', 'low', 'close', 'volume'}

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

    def __str__(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self._strategy_params.items())})"
