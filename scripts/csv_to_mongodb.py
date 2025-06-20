"""
Migration script to move data from CSV files to MongoDB.
Run this to clean up the CSV mess and populate the new MongoDB structure.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
from decimal import Decimal
from typing import List

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from trading_bot.data.market_data import MarketDataManager
from trading_bot.core.models import MarketData
from trading_bot.core.enums import TimeFrame


def find_csv_files(data_dir: str = "data") -> List[Path]:
    """Find all CSV files in the old data directory"""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Data directory {data_dir} does not exist")
        return []
    
    csv_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(Path(root) / file)
    
    return csv_files


def parse_csv_filename(filename: str) -> tuple[str, str, bool]:
    """
    Parse CSV filename to extract symbol, timeframe, and indicator flag.
    Examples:
    - BTCUSDT_15m_since_2023_01_01.csv -> ('BTCUSDT', '15m', False)
    - BTCUSDT_15m_since_2023_01_01_indicators.csv -> ('BTCUSDT', '15m', True)
    """
    parts = filename.replace('.csv', '').split('_')
    
    if len(parts) < 2:
        return None, None, False
    
    symbol = parts[0]
    timeframe = parts[1]
    
    # Convert old timeframe format
    if timeframe == "1Month":
        timeframe = "1M"
    
    has_indicators = 'indicators' in filename
    
    return symbol, timeframe, has_indicators


def migrate_csv_to_mongodb(csv_file: Path, data_manager: MarketDataManager) -> int:
    """Migrate a single CSV file to MongoDB"""
    
    filename = csv_file.name
    symbol, timeframe, has_indicators = parse_csv_filename(filename)
    
    if not symbol or not timeframe:
        print(f"Skipping {filename} - cannot parse symbol/timeframe")
        return 0
    
    # Validate timeframe
    try:
        tf = TimeFrame(timeframe)
    except ValueError:
        print(f"Skipping {filename} - invalid timeframe {timeframe}")
        return 0
    
    print(f"Migrating {symbol} {timeframe} from {filename}")
    
    try:
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Validate required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Skipping {filename} - missing columns: {missing_cols}")
            return 0
        
        # Convert data to MarketData objects
        market_data_list = []
        
        for _, row in df.iterrows():
            try:
                # Handle timestamp conversion
                if isinstance(row['timestamp'], (int, float)):
                    # Timestamp in milliseconds
                    timestamp = datetime.fromtimestamp(row['timestamp'] / 1000)
                else:
                    # Try to parse as datetime string
                    timestamp = pd.to_datetime(row['timestamp']).to_pydatetime()
                
                # Create MarketData object
                market_data = MarketData(
                    symbol=symbol,
                    timeframe=tf,
                    timestamp=timestamp,
                    open=Decimal(str(row['open'])),
                    high=Decimal(str(row['high'])),
                    low=Decimal(str(row['low'])),
                    close=Decimal(str(row['close'])),
                    volume=Decimal(str(row['volume']))
                )
                
                # Add indicators if present
                if has_indicators:
                    indicators = {}
                    for col in df.columns:
                        if col not in required_cols and pd.notna(row[col]):
                            indicators[col] = float(row[col])
                    market_data.indicators = indicators
                
                market_data_list.append(market_data)
                
            except Exception as e:
                print(f"Error processing row in {filename}: {e}")
                continue
        
        if not market_data_list:
            print(f"No valid data found in {filename}")
            return 0
        
        # Store in MongoDB
        stored_count = data_manager.storage.store_candles(market_data_list)
        print(f"✅ Migrated {stored_count}/{len(market_data_list)} candles from {filename}")
        
        return stored_count
        
    except Exception as e:
        print(f"❌ Error migrating {filename}: {e}")
        return 0


def main():
    """Main migration function"""
    
    print("🚀 Starting migration from CSV to MongoDB...")
    
    # Initialize data manager
    try:
        data_manager = MarketDataManager()
        print("✅ Connected to MongoDB")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        return
    
    # Find CSV files
    csv_files = find_csv_files()
    
    if not csv_files:
        print("No CSV files found to migrate")
        return
    
    print(f"Found {len(csv_files)} CSV files to migrate")
    
    # Migrate each file
    total_migrated = 0
    successful_files = 0
    
    for csv_file in csv_files:
        count = migrate_csv_to_mongodb(csv_file, data_manager)
        if count > 0:
            total_migrated += count
            successful_files += 1
    
    print(f"\n📊 Migration Summary:")
    print(f"Files processed: {len(csv_files)}")
    print(f"Successful migrations: {successful_files}")
    print(f"Total candles migrated: {total_migrated}")
    
    # Show database stats
    stats = data_manager.get_database_stats()
    print(f"\n📈 Database Stats:")
    print(f"Total candles in DB: {stats['total_candles']}")
    print(f"Symbols in DB: {stats['symbols']}")
    print(f"Database size: {stats['database_size_mb']:.2f} MB")
    
    # Show per-symbol breakdown
    print(f"\n📋 Symbols in database:")
    for symbol, timeframes in stats['symbols_detail'].items():
        print(f"  {symbol}:")
        for tf, info in timeframes.items():
            start_date = info['date_range'][0].strftime('%Y-%m-%d')
            end_date = info['date_range'][1].strftime('%Y-%m-%d')
            print(f"    {tf}: {info['candles']} candles ({start_date} to {end_date})")
    
    data_manager.close()
    print("\n✅ Migration completed!")


if __name__ == "__main__":
    main()