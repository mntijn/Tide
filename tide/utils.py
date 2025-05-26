from faker import Faker
from typing import Dict, Any, Optional
from .data_structures import AgeGroup

COUNTRY_CODES = [
    "USA", "UK", "JP", "FR", "DE", "NL", "KY", "LI", "MC", "BS", "IM", "AE"
]

# Map to faker
COUNTRY_TO_LOCALE = {
    "USA": "en_US",
    "UK": "en_GB",
    "JP": "ja_JP",
    "FR": "fr_FR",
    "DE": "de_DE",
    "NL": "nl_NL",
    "KY": "en_US",
    "LI": "de_DE",
    "MC": "fr_FR",
    "BS": "en_US",
    "IM": "en_GB",
    "AE": "ar_AE",
}


COUNTRY_TO_CURRENCY = {
    "USA": "USD",
    "UK": "GBP",
    "JP": "JPY",
    "FR": "EUR",
    "DE": "EUR",
    "NL": "EUR",
    "KY": "USD",
    "LI": "CHF",
    "MC": "EUR",
    "BS": "BSD",
    "AE": "AED",
    "IM": "GBP",
}

HIGH_RISK_BUSINESS_CATEGORIES = [
    "Casinos",
    "Money Service Businesses",
    "Precious Metals Dealers",
    "Pawn Shops",
    "Check Cashing Services",
    "Currency Exchange",
    "Virtual Currency Exchange",
    "Art and Antiquities Dealers",
    "Jewelry Stores",
    "Convenience Stores",
    "Gas Stations",
    "Laundromats",
    "Bars and Nightclubs",
    "Auto Dealerships",
    "Used Car Sales",
    "Scrap Metal Dealers",
    "Tobacco Shops",
    "Liquor Stores",
    "Adult Entertainment",
    "Gun Shops",
    "Private Banking",
    "Investment Banking",
    "Trust Services",
]

HIGH_RISK_COUNTRIES = [
    "KY",  # Cayman Islands
    "BS",  # Bahamas
    "CH",  # Switzerland
    "MC",  # Monaco
    "CY",  # Cyprus
    "MT",  # Malta
    "SC",  # Seychelles
    "BB",  # Barbados
    "BM",  # Bermuda
    "AE",  # UAE
    "HK",  # Hong Kong
    "SG",  # Singapore
    "BZ",  # Belize
    "VU",  # Vanuatu
    "AG",  # Antigua and Barbuda
]

HIGH_RISK_OCCUPATIONS = [
    "Banker",
    "Retail banker",
    "Financial adviser",
    "Risk analyst",
    "Risk manager",
    "Corporate investment banker",
    "Investment banker, corporate",
    "Investment banker, operational",
    "Financial trader",

    "Lawyer",
    "Solicitor",
    "Licensed conveyancer",
    "Chartered accountant",
    "Chartered certified accountant",
    "Tax adviser",
    "Tax inspector",

    "Data processing manager",
    "Administrator",
    "Personal assistant",
    "IT consultant",

    "Restaurant manager",
    "Restaurant manager, fast food",
    "Public house manager",
    "Retail manager",
    "Dealer",

    "Estate agent",
    "Jewellery designer",

    "Cabin crew",
    "Tour manager",
    "Youth worker",
    "Community development worker",
    "Student"
]

HIGH_RISK_AGE_GROUPS = [
    "EIGHTEEN_TO_TWENTY_FOUR",
    "SIXTY_FIVE_PLUS"
]


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


def generate_business_category(faker: Faker) -> str:
    """Generate a business category"""
    business_categories = [
        # High-risk/Cash businesses
        "Casinos",
        "Money Service Businesses",
        "Precious Metals Dealers",
        "Pawn Shops",
        "Check Cashing Services",
        "Currency Exchange",
        "Virtual Currency Exchange",
        "Art and Antiquities Dealers",
        "Jewelry Stores",
        "Convenience Stores",
        "Gas Stations",
        "Laundromats",
        "Bars and Nightclubs",
        "Auto Dealerships",
        "Used Car Sales",
        "Scrap Metal Dealers",
        "Tobacco Shops",
        "Liquor Stores",
        "Adult Entertainment",
        "Gun Shops",
        "Private Banking",
        "Investment Banking",
        "Trust Services",

        # Regular businesses
        "Technology Services",
        "Software Development",
        "Consulting Services",
        "Manufacturing",
        "Retail Trade",
        "Food Services",
        "Healthcare Services",
        "Educational Services",
        "Financial Services",
        "Insurance",
        "Real Estate",
        "Construction",
        "Transportation",
        "Logistics",
        "Marketing Agency",
        "Law Firm",
        "Accounting Firm",
        "Engineering Services",
        "Pharmaceutical",
        "Biotechnology",
        "Energy Services",
        "Telecommunications",
        "Media and Entertainment",
        "Publishing",
        "Agriculture",
        "Mining",
        "Utilities",
        "Hospitality",
        "Travel Agency",
        "E-commerce",
        "Import/Export",
        "Wholesale Distribution",
        "Security Services",
        "Cleaning Services",
        "Property Management",
        "Investment Management",
        "Venture Capital",
        "Private Equity",
        "Hedge Fund",
        "Asset Management"
    ]

    return faker.random_element(business_categories)


def map_occupation_to_business_category(occupation: str) -> Optional[str]:
    """Return a plausible business category based on an individual's occupation."""
    occupation_lower = occupation.lower()

    keyword_category_mapping = [
        (["software", "developer", "programmer", "engineer",
         "it", "technology"], "Software Development"),
        (["consultant", "consulting", "management consultant"], "Consulting Services"),
        (["teacher", "professor", "education"], "Educational Services"),
        (["doctor", "nurse", "medical", "physician", "health"], "Healthcare Services"),
        (["lawyer", "attorney", "legal"], "Law Firm"),
        (["accountant", "finance", "accounting"], "Accounting Firm"),
        (["chef", "cook", "restaurant", "food"], "Food Services"),
        (["driver", "transport", "logistics"], "Transportation"),
        (["construction", "builder", "architect", "civil"], "Construction"),
        (["marketing", "advertising", "public relations", "pr"], "Marketing Agency"),
        (["farmer", "agricultur"], "Agriculture"),
        (["sales", "retail"], "Retail Trade"),
        (["entrepreneur", "founder", "owner", "ceo",
         "business owner"], "Management Consulting"),
        (["real estate", "estate agent"], "Real Estate"),
    ]

    for keywords, category in keyword_category_mapping:
        for kw in keywords:
            if kw in occupation_lower:
                return category

    return None


def occupation_indicates_business_owner(occupation: str) -> bool:
    """Determine whether the occupation string suggests the individual could own a business."""
    return map_occupation_to_business_category(occupation) is not None


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
