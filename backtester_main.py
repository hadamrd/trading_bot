from tradingbot2.strategies.VWAPBounceStrategy import VWAPBounceStrategy
from tradingbot2.backtester import Backtester
from tradingbot2.models import BacktestConfig, BacktestResult
from tradingbot2.strategies.MultiFactorStrategy import MultiFactorStrategy
from datetime import datetime
import logging
import csv
import os


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    fh = logging.FileHandler('backtest.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def save_results_to_csv(strategy_params, results: BacktestResult, filename='backtest_results.csv'):
    fieldnames = list(strategy_params.keys()) + [
        'total_profit', 'total_return', 'win_rate', 'total_trades',
        'average_profit_per_trade', 'largest_win', 'largest_loss',
        'max_drawdown', 'sharpe_ratio', 'sortino_ratio', 'profit_factor'
    ]
    
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        row = strategy_params.copy()
        row.update({
            'total_profit': results.total_profit,
            'total_return': (results.total_profit / 3000) * 100,  # Assuming initial balance is always 3000
            'win_rate': results.win_rate,
            'total_trades': results.total_trades,
            'average_profit_per_trade': results.average_profit,
            'largest_win': results.largest_win,
            'largest_loss': results.largest_loss,
            'max_drawdown': results.max_drawdown,
            'sharpe_ratio': results.sharpe_ratio,
            'sortino_ratio': results.sortino_ratio,
            'profit_factor': results.profit_factor
        })
        writer.writerow(row)

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    config = BacktestConfig(
        symbols=['FTMUSDT'],
        timeframe="15m",
        since_date=datetime(2022, 12, 1),
        test_start_date=datetime(2023, 12, 1),
        test_end_date=None,
        price_usdt_rate=1,
        fee_rate=0.001,
        initial_balance=6000
    )

    strategy_params = {
        "vwap_period": 5,
        "rsi_period": 8,
        "rsi_oversold": 40,
        "rsi_overbought": 68,
        "bounce_threshold": 0.0026526772991049326,
        "volume_factor": 1.0034958968836714,
        "take_profit_percentage": 0.04200763886459328,
        "stop_loss_percentage": 0.0226834639412405,
        "atr_period": 21
    }

    # Log strategy parameters
    logger.info("Strategy Parameters:")
    for key, value in strategy_params.items():
        logger.info(f"{key}: {value}")

    strategy = VWAPBounceStrategy(**strategy_params)

    backtester = Backtester(config, strategy)
    results = backtester.run()

    for symbol, result in results.items():
        logger.info(f"\nDetailed Results for {symbol}:")
        logger.info(f"Total Profit: ${result.total_profit:.2f}")
        logger.info(f"Total Return: {(result.total_profit / config.initial_balance) * 100:.2f}%")
        logger.info(f"Win Rate: {result.win_rate:.2%}")
        logger.info(f"Total Trades: {result.total_trades}")
        logger.info(f"Winning Trades: {result.winning_trades}")
        logger.info(f"Losing Trades: {result.losing_trades}")
        logger.info(f"Average Profit per Trade: ${result.average_profit:.2f}")
        logger.info(f"Average Loss per Trade: ${result.average_loss:.2f}")
        logger.info(f"Largest Win: ${result.largest_win:.2f}")
        logger.info(f"Largest Loss: ${result.largest_loss:.2f}")
        logger.info(f"Average Holding Time: {result.average_holding_time:.2f} hours")
        logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"Sortino Ratio: {result.sortino_ratio:.2f}")
        logger.info(f"Max Drawdown: {result.max_drawdown:.2%}")
        logger.info(f"Profit Factor: {result.profit_factor:.2f}")

        logger.info("\nFirst 10 and Last 10 Trades:")
        for trade in result.trades[:10] + result.trades[-10:]:
            logger.info(f"Open: {trade.open_time}, Close: {trade.close_time}, Profit: ${trade.profit:.2f}")

        # Save results to CSV
        save_results_to_csv(strategy_params, result)

if __name__ == "__main__":
    main()