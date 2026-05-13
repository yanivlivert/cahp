import time
from loguru import logger
from functools import wraps
import cahp.config as config

def verbose_decorator(func):
    """
    Decorator to conditionally run a function based on the global verbose flag in config.py.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if config.verbose:
            return func(*args, **kwargs)
    return wrapper


def timing_decorator(func):
    """
    Decorator to measure the time taken to execute a function and log it at the debug level,
    but only if config.verbose is True.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if config.verbose:
            logger.debug(f"{func.__name__} took {elapsed_time:.4f} seconds to execute.")
        return result
    return wrapper
