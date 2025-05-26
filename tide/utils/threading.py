import threading
from queue import Queue
from typing import Callable, List, Tuple, Optional, Any


def run_in_threads(tasks: List[Tuple[Callable, tuple]], join: bool = True) -> None:
    """
    Run a list of (function, args) pairs in parallel threads.
    Example:
        run_in_threads([
            (func1, (arg1,)),
            (func2, (arg2, arg3)),
        ])
    """
    threads = []
    for func, args in tasks:
        t = threading.Thread(target=func, args=args)
        threads.append(t)
        t.start()
    if join:
        for t in threads:
            t.join()


def run_in_threads_with_results(tasks: List[Tuple[Callable, tuple]], results_queue: Optional[Queue] = None) -> List[Any]:
    """
    Run a list of (function, args) pairs in parallel threads, collecting results in a queue.
    Each function should put its result in the queue.
    Returns a list of results (order not guaranteed).
    """
    if results_queue is None:
        results_queue = Queue()
    threads = []
    for func, args in tasks:
        t = threading.Thread(target=func, args=args)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    return results
