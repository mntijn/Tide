from faker import Faker
from ..datastructures.enums import AgeGroup


def generate_age_consistent_occupation(faker: Faker, age_group: AgeGroup) -> str:
    """Return an occupation that is plausible for the given age group.

    For the youngest age group (18-24) we avoid highly specialised or senior
    professions that typically require advanced degrees or lengthy experience
    """

    # Restricted titles for the youngest age bracket
    restricted_keywords = {
        AgeGroup.EIGHTEEN_TO_TWENTY_FOUR: [
            "lawyer", "solicitor", "attorney", "accountant", "chartered",
            "banker", "director", "manager", "physician", "surgeon",
            "doctor", "architect", "partner"
        ]
    }

    occupation = faker.job()

    if age_group in restricted_keywords:
        keywords = restricted_keywords[age_group]
        max_tries = 20
        tries = 0
        while any(k in occupation.lower() for k in keywords) and tries < max_tries:
            occupation = faker.job()
            tries += 1

    return occupation
