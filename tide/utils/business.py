from faker import Faker
from typing import Optional, List, Tuple, Dict
from ..datastructures.enums import AgeGroup


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


def get_max_age_from_group(age_group: AgeGroup) -> int:
    """
    Returns the maximum age for a given age group.
    Used to ensure business creation dates are consistent with owner ages.
    """
    age_mapping = {
        AgeGroup.EIGHTEEN_TO_TWENTY_FOUR: 24,
        AgeGroup.TWENTY_FIVE_TO_THIRTY_FOUR: 34,
        AgeGroup.THIRTY_FIVE_TO_FORTY_NINE: 49,
        AgeGroup.FIFTY_TO_SIXTY_FOUR: 64,
        AgeGroup.SIXTY_FIVE_PLUS: 75
    }
    return age_mapping.get(age_group, 18)


def calculate_age_specific_business_rates(individuals_data: List[Tuple], overall_rate: float, age_probabilities: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate age-specific business creation rates that achieve the overall target rate.
    """
    if not age_probabilities or overall_rate == 0:
        # Fallback to uniform distribution
        return {age_group.value: overall_rate for age_group in [
            data[2]["age_group"] for _, _, data in individuals_data
        ]}

    # Count individuals by age group
    age_group_counts = {}
    for _, _, specific_attrs in individuals_data:
        age_group = specific_attrs["age_group"].value
        age_group_counts[age_group] = age_group_counts.get(age_group, 0) + 1

    total_weighted_individuals = sum(
        age_group_counts.get(age_group, 0) *
        age_probabilities.get(age_group, 0)
        for age_group in age_group_counts.keys()
    )

    total_individuals = sum(age_group_counts.values())

    # Calculate normalization factor to achieve target overall rate
    normalization_factor = (
        overall_rate * total_individuals) / total_weighted_individuals

    # Calculate age-specific rates
    age_specific_rates = {}
    for age_group in age_group_counts.keys():
        weight = age_probabilities.get(age_group, 0)
        age_specific_rates[age_group] = min(1.0, weight * normalization_factor)

    return age_specific_rates
