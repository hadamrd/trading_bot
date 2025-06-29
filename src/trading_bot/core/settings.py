"""
Configuration settings for Trading Bot.
Updated to support both MongoDB and ClickHouse.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global infrastructure settings."""

    # Application
    app_name: str = "Trading Bot"
    version: str = "0.1.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")

    # Database selection
    database_type: str = Field(default="clickhouse", env="DATABASE_TYPE")  # "mongodb" or "clickhouse"

    # MongoDB Configuration (legacy)
    mongodb_url: str = Field(
        default="mongodb://admin:password@localhost:27017/trading_bot?authSource=admin",
        env="MONGODB_URL"
    )
    mongodb_host: str = Field(default="localhost", env="MONGODB_HOST")
    mongodb_port: int = Field(default=27017, env="MONGODB_PORT")
    mongodb_username: str = Field(default="admin", env="MONGODB_USERNAME")
    mongodb_password: str = Field(default="password", env="MONGODB_PASSWORD")
    mongodb_database: str = Field(default="trading_bot", env="MONGODB_DATABASE")
    mongodb_auth_source: str = Field(default="admin", env="MONGODB_AUTH_SOURCE")

    # ClickHouse Configuration (preferred)
    clickhouse_host: str = Field(default="localhost", env="CLICKHOUSE_HOST")
    clickhouse_port: int = Field(default=8123, env="CLICKHOUSE_PORT")  # HTTP port
    clickhouse_username: str = Field(default="default", env="CLICKHOUSE_USERNAME")
    clickhouse_password: str = Field(default="", env="CLICKHOUSE_PASSWORD")
    clickhouse_database: str = Field(default="trading_bot", env="CLICKHOUSE_DATABASE")

    # Binance API Configuration
    binance_api_key: str | None = Field(default=None, env="BINANCE_API_KEY")
    binance_api_secret: str | None = Field(default=None, env="BINANCE_API_SECRET")
    binance_testnet: bool = Field(default=False, env="BINANCE_TESTNET")

    # Rate limiting & timeouts
    api_rate_limit_per_second: int = Field(default=10, env="API_RATE_LIMIT_PER_SECOND")
    api_request_timeout: int = Field(default=30, env="API_REQUEST_TIMEOUT")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str | None = Field(default=None, env="LOG_FILE")
    
    claude_api_key: str | None = Field(default=None, env="CLAUDE_API_KEY")
    claude_model: str = Field(default="claude-3-7-sonnet-20250219", env="CLAUDE_MODEL")

    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get settings singleton instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Convenient alias
settings = get_settings()
