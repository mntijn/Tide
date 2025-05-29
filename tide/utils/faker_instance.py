# Use the same random seed for faker everywhere if specified in graph.yaml

from faker import Faker
import yaml
from pathlib import Path


def get_faker_instance():
    """
    Returns a singleton Faker instance with the random seed from graph.yaml if available.
    """
    if not hasattr(get_faker_instance, '_instance'):
        # Read the config file
        config_path = Path(__file__).parent.parent.parent / \
            'configs' / 'graph.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Get the random seed if it exists, otherwise use None
        seed = config.get('random_seed')

        # Create the Faker instance with the seed
        get_faker_instance._instance = Faker()
        if seed is not None:
            get_faker_instance._instance.seed_instance(seed)

    return get_faker_instance._instance


def reset_faker_seed():
    """Reset the faker instance seed for reproducibility across multiple runs."""
    if hasattr(get_faker_instance, '_instance'):
        config_path = Path(__file__).parent.parent.parent / \
            'configs' / 'graph.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        seed = config.get('random_seed')
        if seed is not None:
            get_faker_instance._instance.seed_instance(seed)


# Create a convenient alias for importing
faker = get_faker_instance()
