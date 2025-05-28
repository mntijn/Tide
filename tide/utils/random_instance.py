# Use the same random seed for random everywhere if specified in graph.yaml

import random
import yaml
from pathlib import Path


def get_random_instance():
    """
    Returns a singleton random instance with the random seed from graph.yaml if available.
    This ensures consistent random number generation across the application.
    """
    if not hasattr(get_random_instance, '_instance'):
        # Read the config file
        config_path = Path(__file__).parent.parent.parent / \
            'configs' / 'graph.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Get the random seed if it exists, otherwise use None
        seed = config.get('random_seed')

        # Create a new random.Random instance
        get_random_instance._instance = random.Random()
        if seed is not None:
            get_random_instance._instance.seed(seed)

    return get_random_instance._instance


# Create a convenient alias for importing
random_instance = get_random_instance()
