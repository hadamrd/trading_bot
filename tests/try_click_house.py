#!/usr/bin/env python3
"""
Debug script to find why data is downloaded but not retrievable
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from trading_bot.data.market_data import MarketDataManager
from trading_bot.core.enums import TimeFrame

def debug_data_storage():
    """Debug the data storage issue"""
    print("🐛 Debugging Data Storage Issue")
    print("=" * 50)
    
    symbol = "SOLUSDT"
    timeframe = TimeFrame.FIFTEEN_MINUTES
    
    try:
        manager = MarketDataManager()
        
        # 1. Check what's in the database
        print(f"\n1️⃣ Checking database contents...")
        total_symbols = manager.storage.get_symbols()
        print(f"   Available symbols: {total_symbols}")
        
        total_count = manager.storage.count_candles(symbol, timeframe)
        print(f"   Total candles for {symbol} {timeframe}: {total_count:,}")
        
        if total_count > 0:
            date_range = manager.storage.get_date_range(symbol, timeframe)
            print(f"   Date range: {date_range}")
        
        # 2. Test raw query
        print(f"\n2️⃣ Testing raw ClickHouse query...")
        query = f"SELECT count(*) FROM market_data WHERE symbol = '{symbol}'"
        result = manager.storage.client.query(query)
        raw_count = result.result_rows[0][0] if result.result_rows else 0
        print(f"   Raw count for {symbol}: {raw_count:,}")
        
        # 3. Check timeframe values
        print(f"\n3️⃣ Checking timeframe values...")
        tf_query = f"SELECT DISTINCT timeframe FROM market_data WHERE symbol = '{symbol}'"
        tf_result = manager.storage.client.query(tf_query)
        stored_timeframes = [row[0] for row in tf_result.result_rows]
        print(f"   Stored timeframes for {symbol}: {stored_timeframes}")
        print(f"   Looking for timeframe: '{timeframe}' (type: {type(timeframe)})")
        
        # 4. Test data retrieval
        print(f"\n4️⃣ Testing data retrieval...")
        df = manager.storage.get_candles_df(symbol, timeframe)
        print(f"   Retrieved DataFrame: {len(df)} rows")
        
        if len(df) > 0:
            print(f"   Columns: {list(df.columns)}")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        
        # 5. Try downloading fresh data
        print(f"\n5️⃣ Testing fresh download...")
        count = manager.download_and_store(symbol, timeframe, days=1, force_update=True)
        print(f"   Downloaded: {count} candles")
        
        # 6. Re-test retrieval after download
        print(f"\n6️⃣ Re-testing retrieval...")
        df2 = manager.storage.get_candles_df(symbol, timeframe)
        print(f"   Retrieved DataFrame after download: {len(df2)} rows")
        
        # 7. Test with different timeframe format
        print(f"\n7️⃣ Testing timeframe string conversion...")
        print(f"   TimeFrame.FIFTEEN_MINUTES = '{TimeFrame.FIFTEEN_MINUTES}'")
        print(f"   str(TimeFrame.FIFTEEN_MINUTES) = '{str(TimeFrame.FIFTEEN_MINUTES)}'")
        print(f"   TimeFrame.FIFTEEN_MINUTES.value = '{TimeFrame.FIFTEEN_MINUTES.value}'")
        
        # Test with explicit string
        df3 = manager.storage.get_candles_df(symbol, "15m")
        print(f"   Retrieved with '15m': {len(df3)} rows")
        
        manager.close()
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

def fix_and_test():
    """Try to fix and test the issue"""
    print(f"\n🔧 ATTEMPTING FIX")
    print("=" * 30)
    
    symbol = "SOLUSDT"
    timeframe = TimeFrame.FIFTEEN_MINUTES
    
    try:
        manager = MarketDataManager()
        
        # Force download small amount of data
        print("1. Downloading small test dataset...")
        count = manager.download_and_store(symbol, timeframe, days=2, force_update=True)
        print(f"   Downloaded: {count} candles")
        
        # Immediately test retrieval
        print("2. Testing immediate retrieval...")
        df = manager.get_data_for_backtest(
            symbol=symbol,
            timeframe=timeframe,
            start_date=datetime.now() - timedelta(days=1),
            with_indicators=False
        )
        
        if len(df) > 0:
            print(f"✅ SUCCESS! Retrieved {len(df)} candles")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            return True
        else:
            print(f"❌ Still no data retrieved")
            return False
            
    except Exception as e:
        print(f"❌ Fix attempt failed: {e}")
        return False

def main():
    """Main debug function"""
    print("🚀 Data Storage Debugger")
    
    # Debug the issue
    debug_data_storage()
    
    # Try to fix
    if fix_and_test():
        print(f"\n🎉 Issue appears to be fixed!")
        print(f"You can now run your backtest again.")
    else:
        print(f"\n❌ Issue persists. Manual intervention needed.")
        print(f"\n💡 Possible solutions:")
        print(f"   1. Restart ClickHouse: docker restart trading-clickhouse")
        print(f"   2. Check ClickHouse logs: docker logs trading-clickhouse")
        print(f"   3. Recreate tables: DROP TABLE market_data; then restart")

if __name__ == "__main__":
    main()