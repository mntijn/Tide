from ..utils.faker_instance import get_faker_instance
from typing import Dict, Any
from .constants import COUNTRY_TO_LOCALE
from faker import Faker
import yaml
from pathlib import Path


def get_seed_from_config():
    """Get the random seed from the config file if it exists."""
    config_path = Path(__file__).parent.parent.parent / \
        'configs' / 'graph.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('random_seed')


def generate_localized_address(country_code: str) -> Dict[str, Any]:
    """Generate an address localized to the country_code"""
    locale = COUNTRY_TO_LOCALE.get(country_code)
    seed = get_seed_from_config()

    try:
        # Create a new Faker instance with the specific locale
        localized_faker = Faker(locale)
        # Set the seed if it exists in the config
        if seed is not None:
            localized_faker.seed_instance(seed)
    except AttributeError:
        localized_faker = Faker("nl_NL")
        if seed is not None:
            localized_faker.seed_instance(seed)

    return {
        "street_address": localized_faker.street_address(),
        "city": localized_faker.city(),
        "country": country_code,
        "postcode": localized_faker.postcode()
    }
