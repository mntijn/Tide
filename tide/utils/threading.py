import threading
from queue import Queue
from typing import Callable, List, Tuple, Optional, Any
import random
from .random_instance import get_random_instance


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


def run_patterns_in_parallel(pattern_tasks: List[Tuple[Callable, tuple, int]], base_seed: Optional[int] = None) -> List[Tuple[str, str, Any]]:
    """
    Run pattern injection tasks with deterministic seeding.

    For now, runs sequentially to ensure determinism since patterns use the global
    random module extensively. To make this truly parallel, patterns would need to
    be modified to accept a random instance parameter.

    Args:
        pattern_tasks: List of (pattern_function, args, task_index) tuples
        base_seed: Base seed for deterministic random number generation

    Returns:
        List of edges returned by each pattern
    """
    results = []

    # Sort by task_index to ensure deterministic order
    sorted_tasks = sorted(pattern_tasks, key=lambda x: x[2])

    for pattern_func, args, task_index in sorted_tasks:
        if base_seed is not None:
            # Set deterministic seed for this specific pattern
            pattern_seed = base_seed + task_index + 1000
            random.seed(pattern_seed)
            # Also seed the global random module to ensure complete determinism
            import random as global_random
            global_random.seed(pattern_seed)

        try:
            edges = pattern_func(*args)
            results.append(edges)
        except Exception as e:
            results.append([])  # Return empty list on error

    return results
