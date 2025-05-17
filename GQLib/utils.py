import time
import logging
logging.getLogger(__name__)

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        logging.info(f"Execution time: {end_time - start_time:.2f} seconds")
        return result
    return wrapper