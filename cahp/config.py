import os

from loguru import logger
import sys
from datetime import datetime

# parameters
verbose = False


# functions
def init_logger():
    """
    Initializes the logger to output to both the console and a timestamped file.
    """
    
    logger.remove()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_folder = os.path.join('outputs', timestamp)
    os.makedirs(log_folder, exist_ok=True)

    # Set up logging to the console
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{function: <20}</cyan> | "
               "<level>{message}</level>",
        level="TRACE"
    )

    # Set up logging to the file
    log_filename = os.path.join(log_folder, 'log.log')
    logger.add(
        log_filename,
        format="{time:YYYY-MM-DD HH:mm:ss} | "
               "{level: <8} | "
               "{function: <20} | "
               "{message}",
        level="TRACE"
    )
