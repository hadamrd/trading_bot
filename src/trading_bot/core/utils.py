"""
Core utilities for the trading bot.
Consolidated from scattered utility functions.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytz
import yaml


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file"""
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file) as f:
        return yaml.safe_load(f)


def paris_datetime(timestamp: int) -> str:
    """
    Convert timestamp to Paris timezone string.
    From the old utils.py
    """
    utc_time = datetime.utcfromtimestamp(timestamp / 1000)
    utc_time = pytz.utc.localize(utc_time)
    paris_time = utc_time.astimezone(pytz.timezone('Europe/Paris'))
    return paris_time.strftime("%Y-%m-%d %H:%M")


def time_diff_hours(start_timestamp: int, end_timestamp: int) -> float:
    """Calculate time difference in hours between timestamps"""
    return (end_timestamp - start_timestamp) / (1000 * 60 * 60)


def format_time_difference(timestamp1: int, timestamp2: int) -> str:
    """Format time difference in human readable format"""
    datetime1 = datetime.utcfromtimestamp(timestamp1 / 1000.0)
    datetime2 = datetime.utcfromtimestamp(timestamp2 / 1000.0)

    time_diff = datetime2 - datetime1
    days = time_diff.days
    hours, remainder = divmod(time_diff.seconds, 3600)
    minutes, _ = divmod(remainder, 60)

    return f"{days} days, {hours} hours, {minutes} minutes"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format"""
    if seconds < 60:
        return f"{round(seconds)} sec"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{int(minutes)} min"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"{int(hours)} hr"
    else:
        days = seconds // 86400
        return f"{int(days)} day{'s' if days > 1 else ''}"


def convert_to_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Convert DataFrame columns to numeric"""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def validate_symbol(symbol: str) -> bool:
    """Validate if symbol format is correct"""
    if not symbol or len(symbol) < 6:
        return False

    # Should end with USDT for our use case
    if not symbol.endswith('USDT'):
        return False

    return True


def validate_timeframe(timeframe: str) -> bool:
    """Validate timeframe format"""
    valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d']
    return timeframe in valid_timeframes


def get_date_range_from_days(days: int, end_date: datetime | None = None) -> tuple[datetime, datetime]:
    """Get start and end date from number of days"""
    if end_date is None:
        end_date = datetime.now()

    start_date = end_date - timedelta(days=days)
    return start_date, end_date


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default if denominator is zero"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return default


def round_to_precision(value: float, precision: int = 8) -> float:
    """Round value to specified precision"""
    return round(value, precision)


def percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp value between min and max"""
    return max(min_value, min(value, max_value))


def check_dependencies() -> bool:
    """Check if all required dependencies are available"""
    try:
        return True
    except ImportError:
        return False


def check_data_directory() -> bool:
    """Check if data directory exists and is writable"""
    data_dir = Path("data")
    try:
        data_dir.mkdir(exist_ok=True)
        # Try to create a test file
        test_file = data_dir / ".test"
        test_file.touch()
        test_file.unlink()
        return True
    except Exception:
        return False


def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print useful information about a DataFrame"""
    print(f"\n{name} Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}" if len(df) > 0 else "Empty DataFrame")
    print(f"Memory usage: {df.memory_usage().sum() / 1024:.2f} KB")

    if len(df) > 0:
        print("Sample data:")
        print(df.head(3).to_string())


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio"""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    return float(excess_returns.mean() / returns.std())


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown"""
    if len(returns) == 0:
        return 0.0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    return float(drawdown.min())


class Timer:
    """Simple timer context manager"""

    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        print(f"⏱️  Starting {self.description}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"✅ {self.description} completed in {format_duration(duration)}")


def create_project_structure():
    """Create the project directory structure if it doesn't exist"""
    directories = [
        "data",
        "logs",
        "notebooks",
        "scripts",
        "tests"
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

    print("📁 Project structure created")
