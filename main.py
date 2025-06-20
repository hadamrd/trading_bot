import logging
from pathlib import Path

from tradingbot2.genetic_optimizer import GeneticOptimizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('optimization.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        config_path = Path("config/optimization_config.json")
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        optimizer = GeneticOptimizer(config_path)
        result = optimizer.run()
        
        # Log summary of results
        logger.info("\nOptimization Results:")
        logger.info(f"Strategy: {result.strategy_name}")
        logger.info(f"Best Fitness: {result.best_fitness}")
        logger.info("\nBest Parameters:")
        for key, value in result.best_parameters.items():
            logger.info(f"{key}: {value}")
        logger.info(f"\nTotal Runtime: {result.run_info['duration_seconds']:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()