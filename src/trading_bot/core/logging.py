"""
Professional logging system for the trading bot
Replaces all print statements with configurable logging
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from ..core.settings import get_settings


class TradingBotFormatter(logging.Formatter):
    """Custom formatter with colors and emojis for better readability"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    # Emojis for different components
    COMPONENT_EMOJIS = {
        'optimization': '🧬',
        'backtest': '🧪',
        'data': '📊',
        'strategy': '🎯',
        'analysis': '📈',
        'engine': '⚙️',
        'storage': '💾',
        'genetic': '🧬',
        'binance': '🔗'
    }
    
    def format(self, record):
        # Add color to levelname
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, '')
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"
        
        # Add emoji based on logger name
        emoji = ''
        for component, component_emoji in self.COMPONENT_EMOJIS.items():
            if component in record.name.lower():
                emoji = f"{component_emoji} "
                break
        
        # Format the message
        original_format = super().format(record)
        return f"{emoji}{original_format}"


class LoggingConfig:
    """Central logging configuration"""
    
    # Define log levels for different components
    DEFAULT_LEVELS = {
        'root': 'INFO',
        'trading_bot': 'INFO',
        'trading_bot.optimization': 'INFO',
        'trading_bot.backtesting': 'WARNING',  # Reduce noise during optimization
        'trading_bot.data': 'WARNING',         # Reduce data download spam
        'trading_bot.strategies': 'WARNING',   # Strategy execution details
        'trading_bot.analysis': 'INFO'
    }
    
    # Optimization mode - much quieter
    OPTIMIZATION_LEVELS = {
        'root': 'WARNING',
        'trading_bot': 'WARNING', 
        'trading_bot.optimization': 'INFO',    # Keep optimization progress
        'trading_bot.backtesting': 'ERROR',    # Only errors
        'trading_bot.data': 'ERROR',           # Only errors
        'trading_bot.strategies': 'ERROR',     # Only errors
        'trading_bot.analysis': 'WARNING'
    }
    
    # Debug mode - everything
    DEBUG_LEVELS = {
        'root': 'DEBUG',
        'trading_bot': 'DEBUG',
        'trading_bot.optimization': 'DEBUG',
        'trading_bot.backtesting': 'DEBUG',
        'trading_bot.data': 'DEBUG',
        'trading_bot.strategies': 'DEBUG',
        'trading_bot.analysis': 'DEBUG'
    }
    
    @classmethod
    def setup_logging(cls, 
                     mode: str = 'default',
                     console_output: bool = True,
                     file_output: bool = True,
                     log_file: Optional[str] = None):
        """
        Setup logging configuration
        
        Args:
            mode: 'default', 'optimization', 'debug', or 'silent'
            console_output: Whether to log to console
            file_output: Whether to log to file
            log_file: Custom log file path
        """
        
        # Choose log levels based on mode
        if mode == 'optimization':
            levels = cls.OPTIMIZATION_LEVELS
        elif mode == 'debug':
            levels = cls.DEBUG_LEVELS
        elif mode == 'silent':
            levels = {logger: 'CRITICAL' for logger in cls.DEFAULT_LEVELS}
        else:
            levels = cls.DEFAULT_LEVELS
        
        # Clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Create formatters
        console_formatter = TradingBotFormatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if console_output and mode != 'silent':
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if file_output:
            if log_file is None:
                log_dir = Path("logs")
                log_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = log_dir / f"trading_bot_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        # Set levels for each logger
        for logger_name, level in levels.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(getattr(logging, level))
        
        # Set root level to the most permissive
        min_level = min(getattr(logging, level) for level in levels.values())
        root_logger.setLevel(min_level)
        
        return log_file


# Convenience functions for different modes
def setup_default_logging():
    """Setup default logging - good for manual backtests"""
    return LoggingConfig.setup_logging('default')


def setup_optimization_logging():
    """Setup optimization logging - minimal noise during GA"""
    return LoggingConfig.setup_logging('optimization', console_output=True)


def setup_debug_logging():
    """Setup debug logging - everything visible"""
    return LoggingConfig.setup_logging('debug')


def setup_silent_logging():
    """Setup silent logging - only critical errors"""
    return LoggingConfig.setup_logging('silent')


# Logger factory with component-specific loggers
def get_logger(name: str = None, component: str = None) -> logging.Logger:
    """
    Get a logger for a specific component
    
    Args:
        name: Logger name (usually __name__)
        component: Component type for emoji selection
        
    Returns:
        Configured logger
    """
    if name is None:
        name = 'trading_bot'
    
    logger = logging.getLogger(name)
    
    # Add component info for formatter
    if component and not hasattr(logger, '_component'):
        logger._component = component
    
    return logger


# Environment-based setup
def setup_logging_from_env():
    """Setup logging based on environment variables"""
    import os
    
    mode = os.getenv('TRADING_BOT_LOG_MODE', 'default').lower()
    console = os.getenv('TRADING_BOT_LOG_CONSOLE', 'true').lower() == 'true'
    file_output = os.getenv('TRADING_BOT_LOG_FILE', 'true').lower() == 'true'
    log_file = os.getenv('TRADING_BOT_LOG_PATH')
    
    return LoggingConfig.setup_logging(
        mode=mode,
        console_output=console, 
        file_output=file_output,
        log_file=log_file
    )


# Context manager for temporary logging changes
class LoggingContext:
    """Context manager to temporarily change logging levels"""
    
    def __init__(self, mode: str):
        self.mode = mode
        self.original_levels = {}
    
    def __enter__(self):
        # Store current levels
        for logger_name in LoggingConfig.DEFAULT_LEVELS:
            logger = logging.getLogger(logger_name)
            self.original_levels[logger_name] = logger.level
        
        # Apply new levels
        if self.mode == 'optimization':
            levels = LoggingConfig.OPTIMIZATION_LEVELS
        elif self.mode == 'debug':
            levels = LoggingConfig.DEBUG_LEVELS
        elif self.mode == 'silent':
            levels = {name: 'CRITICAL' for name in LoggingConfig.DEFAULT_LEVELS}
        else:
            levels = LoggingConfig.DEFAULT_LEVELS
            
        for logger_name, level in levels.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(getattr(logging, level))
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original levels
        for logger_name, level in self.original_levels.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)


# Example usage in modules:
"""
# At the top of any module, replace prints with:
import logging
logger = logging.getLogger(__name__)

# Instead of: print(f"Starting optimization...")
logger.info("Starting optimization...")

# Instead of: print(f"❌ Error: {e}")  
logger.error("Error occurred: %s", e)

# Instead of: print(f"✅ Downloaded {count} candles")
logger.info("Downloaded %d candles", count)

# For debug info:
logger.debug("Processing row: %s", row)
"""