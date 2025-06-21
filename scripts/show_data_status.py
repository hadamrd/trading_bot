#!/usr/bin/env python3
"""
Quick script to check what data is available in your database.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from trading_bot.data.market_data import MarketDataManager
from trading_bot.core.enums import TimeFrame


def show_data_status():
    """Show what data is currently available."""
    
    print("📊 Database Status")
    print("=" * 50)
    
    try:
        dm = MarketDataManager()
        
        # Get overall stats
        stats = dm.get_database_stats()
        print(f"Total candles: {stats['total_candles']:,}")
        print(f"Symbols: {stats['symbols']}")
        print(f"Database size: {stats['database_size_mb']:.2f} MB")
        
        print(f"\n📋 Available Data:")
        print("-" * 30)
        
        symbols = dm.get_available_symbols()
        
        if not symbols:
            print("No data found in database")
            return
        
        for symbol in sorted(symbols):
            # Check for 15m data (most common)
            try:
                info = dm.get_data_info(symbol, TimeFrame.FIFTEEN_MINUTES)
                
                if info['date_range']:
                    start_date = info['date_range'][0].strftime('%Y-%m-%d')
                    end_date = info['date_range'][1].strftime('%Y-%m-%d')
                    candles = info['candle_count']
                    
                    print(f"{symbol:12} | {candles:>6,} candles | {start_date} to {end_date}")
                else:
                    print(f"{symbol:12} | No 15m data")
                    
            except Exception as e:
                print(f"{symbol:12} | Error: {e}")
        
        dm.close()
        
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")


def main():
    """Main function."""
    show_data_status()


if __name__ == "__main__":
    main()