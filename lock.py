import time
import os
from contextlib import contextmanager


@contextmanager
def file_lock(path):
    """Create a non-atomic file lock that works across machines."""
    while os.path.exists(path):
        time.sleep(1)
    with open(path, 'a') as f:
        f.write(f'{os.getpid()}\n')
    yield
    os.remove(path)
