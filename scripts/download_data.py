"""
Script to download market data from Binance and store in MongoDB.
Use this to populate your database with fresh data.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from trading_bot.data.market_data import MarketDataManager
from trading_bot.core.enums import TimeFrame


def download_single_symbol(symbol: str, timeframe: str, days: int):
    """Download data for a single symbol"""
    
    try:
        tf = TimeFrame(timeframe)
    except ValueError:
        print(f"❌ Invalid timeframe: {timeframe}")
        print(f"Valid timeframes: {[tf.value for tf in TimeFrame]}")
        return
    
    print(f"📥 Downloading {symbol} {timeframe} for {days} days...")
    
    try:
        dm = MarketDataManager()
        count = dm.download_and_store(symbol, tf, days)
        
        if count > 0:
            print(f"✅ Downloaded {count} candles for {symbol}")
            
            # Calculate indicators
            print(f"🔢 Calculating indicators...")
            indicator_count = dm.update_indicators(symbol, tf)
            print(f"✅ Updated indicators for {indicator_count} candles")
            
            # Show data info
            info = dm.get_data_info(symbol, tf)
            start_date = info['date_range'][0].strftime('%Y-%m-%d %H:%M')
            end_date = info['date_range'][1].strftime('%Y-%m-%d %H:%M')
            print(f"📊 Data range: {start_date} to {end_date}")
            print(f"💰 Latest price: ${info['latest_price']:.4f}")
        else:
            print(f"ℹ️  No new data downloaded for {symbol}")
        
        dm.close()
        
    except Exception as e:
        print(f"❌ Error downloading {symbol}: {e}")


def download_multiple_symbols(symbols: list, timeframe: str, days: int):
    """Download data for multiple symbols"""
    
    try:
        tf = TimeFrame(timeframe)
    except ValueError:
        print(f"❌ Invalid timeframe: {timeframe}")
        return
    
    print(f"📥 Downloading {len(symbols)} symbols...")
    
    try:
        dm = MarketDataManager()
        results = dm.bulk_download(symbols, tf, days)
        
        print(f"\n📊 Download Summary:")
        total_candles = 0
        successful = 0
        
        for symbol, count in results.items():
            if count > 0:
                print(f"✅ {symbol}: {count} candles")
                total_candles += count
                successful += 1
                
                # Update indicators
                dm.update_indicators(symbol, tf)
            else:
                print(f"⚠️  {symbol}: No new data")
        
        print(f"\nSuccessful downloads: {successful}/{len(symbols)}")
        print(f"Total candles downloaded: {total_candles}")
        
        dm.close()
        
    except Exception as e:
        print(f"❌ Error during bulk download: {e}")


def get_popular_symbols():
    """Get list of popular trading symbols"""
    return [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT",
        "XRPUSDT", "DOTUSDT", "DOGEUSDT", "AVAXUSDT", "MATICUSDT",
        "LINKUSDT", "LTCUSDT", "UNIUSDT", "ATOMUSDT", "FTMUSDT",
        "NEARUSDT", "ALGOUSDT", "MANAUSDT", "SANDUSDT", "AXSUSDT"
    ]


def main():
    parser = argparse.ArgumentParser(description="Download market data from Binance")
    
    # Symbol options
    symbol_group = parser.add_mutually_exclusive_group(required=False)
    symbol_group.add_argument(
        "--symbol", "-s", 
        type=str, 
        help="Single symbol to download (e.g., BTCUSDT)"
    )
    symbol_group.add_argument(
        "--symbols", 
        nargs="+", 
        help="Multiple symbols to download"
    )
    symbol_group.add_argument(
        "--popular", 
        action="store_true", 
        help="Download popular symbols"
    )
    symbol_group.add_argument(
        "--all-usdt",
        action="store_true",
        help="Download all USDT pairs"
    )
    
    # Other options
    parser.add_argument(
        "--timeframe", "-t",
        default="15m",
        choices=[tf.value for tf in TimeFrame],
        help="Timeframe to download (default: 15m)"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=30,
        help="Number of days to download (default: 30)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.stats and not any([args.symbol, args.symbols, args.popular, args.all_usdt]):
        parser.error("Must specify either --stats or one of --symbol/--symbols/--popular/--all-usdt")
    
    # Show stats if requested
    if args.stats:
        try:
            dm = MarketDataManager()
            stats = dm.get_database_stats()
            
            print(f"📊 Database Statistics:")
            print(f"Total candles: {stats['total_candles']:,}")
            print(f"Number of symbols: {stats['symbols']}")
            print(f"Database size: {stats['database_size_mb']:.2f} MB")
            
            print(f"\n📋 Available symbols:")
            for symbol in sorted(dm.get_available_symbols()):
                print(f"  {symbol}")
            
            dm.close()
            return
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return
    
    # Download data
    if args.symbol:
        download_single_symbol(args.symbol.upper(), args.timeframe, args.days)
    
    elif args.symbols:
        symbols = [s.upper() for s in args.symbols]
        download_multiple_symbols(symbols, args.timeframe, args.days)
    
    elif args.popular:
        symbols = get_popular_symbols()
        print(f"📈 Downloading popular symbols: {', '.join(symbols)}")
        download_multiple_symbols(symbols, args.timeframe, args.days)
    
    elif args.all_usdt:
        try:
            from trading_bot.data.binance_client import BinanceClient
            client = BinanceClient()
            symbols = client.get_usdt_pairs()
            print(f"💰 Found {len(symbols)} USDT pairs")
            download_multiple_symbols(symbols, args.timeframe, args.days)
        except Exception as e:
            print(f"❌ Error getting USDT pairs: {e}")


if __name__ == "__main__":
    main()