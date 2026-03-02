import logging
import threading
from queue import Queue
from typing import Any, Callable

logger = logging.getLogger(__name__)


def run_in_threads(tasks: list[tuple[Callable, tuple]], join: bool = True) -> None:
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


def run_in_threads_with_results(tasks: list[tuple[Callable, tuple]], results_queue: Queue | None = None) -> list[Any]:
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


def run_patterns_in_parallel(pattern_tasks: list[tuple[Callable, tuple, int]]) -> list[tuple[str, str, Any]]:
    """
    Run pattern injection tasks with deterministic seeding.

    For now, runs sequentially to ensure determinism since patterns use the global
    random module extensively. To make this truly parallel, patterns would need to
    be modified to accept a random instance parameter.

    The global random module is already seeded in random_instance.py, so we don't
    need to re-seed it here.

    Args:
        pattern_tasks: list of (pattern_function, args, task_index) tuples

    Returns:
        list of edges returned by each pattern
    """
    results = []

    # Sort by task_index to ensure deterministic order
    sorted_tasks = sorted(pattern_tasks, key=lambda x: x[2])

    for pattern_func, args, task_index in sorted_tasks:
        try:
            logger.debug("Running pattern %s", pattern_func)
            edges = pattern_func(*args)
            logger.debug("Pattern %s completed", pattern_func)
            results.append(edges)
        except Exception as e:
            logger.error("Pattern %s failed: %s", pattern_func, e)
            results.append([])

    return results
