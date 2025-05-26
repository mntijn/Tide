from faker import Faker
from typing import Dict, Any
from .constants import COUNTRY_TO_LOCALE


def generate_localized_address(country_code: str) -> Dict[str, Any]:
    """Generate an address localized to the country_code"""
    locale = COUNTRY_TO_LOCALE.get(country_code)
    try:
        localized_faker = Faker(locale)
    except AttributeError:
        localized_faker = Faker("nl_NL")

    return {
        "street_address": localized_faker.street_address(),
        "city": localized_faker.city(),
        "country": country_code,
        "postcode": localized_faker.postcode()
    }
