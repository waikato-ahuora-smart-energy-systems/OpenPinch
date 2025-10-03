import logging
from time import perf_counter as timer
from functools import wraps
import atexit
from collections import defaultdict
from ..lib import *

# Configure logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("timing.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

_function_stats = defaultdict(lambda: {"count": 0, "total_time": 0.0})

def timing_decorator(func=None, *, activate_overide=False):
    """
    Decorator to measure execution time and track per-function totals.
    Supports both @timing_decorator and @timing_decorator(activate_overide=True)
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not (config.ACTIVATE_TIMING or activate_overide):
                return f(*args, **kwargs)

            start_time = timer()
            result = f(*args, **kwargs)
            end_time = timer()
            exec_time = end_time - start_time

            stats = _function_stats[f.__name__]
            stats["count"] += 1
            stats["total_time"] += exec_time

            if config.LOG_TIMING:
                logger.info(f"Function '{f.__name__}' executed in {exec_time:.6f} seconds.")

            return result
        return wrapper

    # Handle both @timing_decorator and @timing_decorator(activate_overide=True)
    if callable(func):
        return decorator(func)
    return decorator

@atexit.register
def print_summary():
    logger.info("==== Execution Time Summary ====")
    for func_name, stats in _function_stats.items():
        avg_time = stats["total_time"] / stats["count"]
        logger.info(
            f"{func_name}: {stats['count']} calls, "
            f"total {stats['total_time']:.6f}s, "
            f"average {avg_time:.6f}s"
        )
