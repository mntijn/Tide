# Use the same random seed for random everywhere if specified in graph.yaml
#
# Best practice for reproducibility:
# - Use `random_instance` for scalar random operations (choice, randint, uniform, etc.)
# - Use `numpy_rng` for vectorized random operations (arrays of random numbers)
# - NEVER use the global `np.random` or `random` modules directly
#
# Both are seeded from the same config and reset together via reset_random_seed().

import random
import yaml
import numpy as np
from pathlib import Path

# Cache the seed value
_cached_seed = None

# Dedicated NumPy Generator for vectorized operations
# Using numpy.random.Generator (modern API) for better reproducibility
_numpy_rng = None


def _get_seed_from_config():
    """Read the random seed from graph.yaml config file."""
    global _cached_seed
    config_path = (
        Path(__file__).parent.parent.parent / 'configs' / 'graph.yaml'
    )
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    _cached_seed = config.get('random_seed')
    return _cached_seed


def get_random_instance():
    """
    Returns a singleton random instance with the random seed from graph.yaml if available.
    This ensures consistent random number generation across the application.

    Use this for scalar random operations (single values).
    For vectorized operations (arrays), use get_numpy_rng() instead.
    """
    if not hasattr(get_random_instance, '_instance'):
        seed = _get_seed_from_config()

        # Create a new random.Random instance
        get_random_instance._instance = random.Random()
        if seed is not None:
            get_random_instance._instance.seed(seed)
            # Also seed the global random module for complete determinism
            random.seed(seed)

    return get_random_instance._instance


def get_numpy_rng():
    """
    Returns a singleton NumPy Generator for vectorized random operations.

    Use this for operations that generate arrays of random numbers.
    For scalar operations, use get_random_instance() instead.

    The Generator API (numpy.random.Generator) is the modern NumPy random API
    and provides better statistical properties and reproducibility than the
    legacy np.random functions.
    """
    global _numpy_rng
    if _numpy_rng is None:
        seed = _get_seed_from_config()
        if seed is not None:
            _numpy_rng = np.random.Generator(np.random.PCG64(seed))
        else:
            _numpy_rng = np.random.Generator(np.random.PCG64())
    return _numpy_rng


def reset_random_seed(seed=None):
    """
    Reset all random states for reproducibility across multiple runs.

    This function MUST be called at the start of each graph generation
    to ensure deterministic behavior when running multiple times in
    the same Python process.

    Args:
        seed: Optional seed value. If None, reads from graph.yaml config.
    """
    global _cached_seed, _numpy_rng

    if seed is None:
        seed = _get_seed_from_config()

    _cached_seed = seed

    if seed is not None:
        # Reset the singleton random instance
        if hasattr(get_random_instance, '_instance'):
            get_random_instance._instance.seed(seed)

        # Reset global Python random module
        random.seed(seed)

        # Reset the NumPy Generator with fresh state
        _numpy_rng = np.random.Generator(np.random.PCG64(seed))
    else:
        # No seed - create unseeded generator
        _numpy_rng = np.random.Generator(np.random.PCG64())


# Create convenient alias for importing
# NOTE: For numpy_rng, always use get_numpy_rng() to ensure you get the
# current generator after reset_random_seed() is called.
random_instance = get_random_instance()
