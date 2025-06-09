from typing import Dict, Any, Tuple, Optional, List
import datetime
from ..datastructures.enums import AgeGroup
from ..datastructures.attributes import NodeAttributes
from ..utils.constants import (
    HIGH_RISK_BUSINESS_CATEGORIES, HIGH_RISK_COUNTRIES,
    COUNTRY_CODES, HIGH_RISK_OCCUPATIONS, HIGH_PAID_OCCUPATIONS
)
from ..utils.business import (
    generate_business_category,
    get_random_age_from_group
)
from .base import Entity


class Business(Entity):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)

    def _calculate_risk_score(self, specific_attrs: Dict[str, Any], common_attrs: Dict[str, Any]) -> float:
        """Calculates a risk score for a business."""
        score = self._calculate_base_risk_score(common_attrs)

        # Add risk score for business category
        business_category = specific_attrs.get("business_category")
        if business_category in HIGH_RISK_BUSINESS_CATEGORIES:
            category_weight = self.risk_weights.get("business_category", 0.30)
            category_factor = self.risk_config.get(
                "business_categories_weight_factor", 1.0)
            score += category_weight * category_factor

        # Add risk score for company location
        company_country = common_attrs.get("address", {}).get("country")
        if company_country in HIGH_RISK_COUNTRIES:
            country_weight = self.risk_weights.get("country", 0.25)
            country_factor = self.risk_config.get(
                "countries_weight_factor", 1.0)
            score += country_weight * country_factor

        # Add risk score for company size
        num_employees = specific_attrs.get("number_of_employees", 0)
        company_size_thresholds = self.risk_config.get(
            "company_size_thresholds", {})

        if num_employees <= company_size_thresholds.get("very_small_max", 5):
            score += self.risk_weights.get("very_small_company", 0.05)

        max_score = self.risk_weights.get("max_score", 0.9)
        return min(score, max_score)

    def _calculate_offshore_probability(
        self,
        owner_occupation: str,
        owner_risk_score: float,
        business_category: str,
        owner_country: str
    ) -> float:
        """Calculate the probability of a business being registered in a different country than its owner."""
        # Start with base probability from config
        probability = self.params.get("high_risk_business_probability", 0.05)

        # Increase probability based on owner's risk score
        probability += owner_risk_score * 0.1

        # Increase probability for high-risk business categories
        if business_category in HIGH_RISK_BUSINESS_CATEGORIES:
            probability += 0.15

        # Increase probability for certain occupations that suggest international business
        if owner_occupation in HIGH_RISK_OCCUPATIONS:
            probability += 0.1
        if owner_occupation in HIGH_PAID_OCCUPATIONS:
            probability += 0.15

        # Increase probability if owner is in a high-risk country
        if owner_country in HIGH_RISK_COUNTRIES:
            probability += 0.1

        return min(probability, 0.4)

    def generate_data(self) -> List[Tuple[datetime.datetime, Dict[str, Any], Dict[str, Any]]]:
        """Generates data for business nodes, including a risk score."""
        businesses_data = []
        num_businesses = self.graph_scale.get("businesses", 0)
        start_date = self.time_span["start_date"]

        company_size_range = self.params.get("company_size_range")
        business_creation_range = self.params.get(
            "business_creation_date_range", [90, 5475])

        for _ in range(num_businesses):
            creation_date = start_date - datetime.timedelta(
                days=self.random_instance.randint(business_creation_range[0], business_creation_range[1]))
            country_code = self.random_instance.choice(COUNTRY_CODES)
            business_category = generate_business_category(self.faker)

            is_in_high_risk_category = business_category in HIGH_RISK_BUSINESS_CATEGORIES
            is_in_high_risk_country = country_code in HIGH_RISK_COUNTRIES

            common_attrs = {
                "country_code": country_code,
                "is_fraudulent": False
            }
            specific_attrs = {
                "name": self.faker.company(),
                "business_category": business_category,
                "incorporation_year": creation_date.year,
                "number_of_employees": self.random_instance.randint(company_size_range[0], company_size_range[1]),
                "is_high_risk_category": is_in_high_risk_category,
                "is_high_risk_country": is_in_high_risk_country,
            }
            risk_score = self._calculate_risk_score(
                specific_attrs, common_attrs)
            common_attrs["risk_score"] = risk_score

            businesses_data.append(
                (creation_date, common_attrs, specific_attrs))
        return businesses_data

    def generate_age_consistent_business_for_individual(
        self,
        individual_age_group: AgeGroup,
        sim_start_date: datetime.datetime,
        business_category_override: Optional[str] = None,
        owner_occupation: Optional[str] = None,
        owner_risk_score: Optional[float] = None,
        owner_country: Optional[str] = None,
    ) -> Tuple[datetime.datetime, Dict[str, Any], Dict[str, Any]]:
        """Generates a business that is consistent with the given individual's age."""
        # Pick a random age within the person's age group
        individual_age = get_random_age_from_group(individual_age_group)

        # Calculate when this person would have turned 18
        years_since_18 = individual_age - 18
        earliest_business_date = sim_start_date - \
            datetime.timedelta(days=years_since_18 * 365)

        # Get business creation date constraints from params
        business_creation_range = self.params.get(
            "business_creation_date_range", [90, 5475])
        latest_business_date = sim_start_date - \
            datetime.timedelta(days=business_creation_range[0])
        oldest_possible_business = sim_start_date - \
            datetime.timedelta(days=business_creation_range[1])

        # Business can't be created before person turned 18 or before the oldest allowed business date
        min_business_date = max(earliest_business_date,
                                oldest_possible_business)
        max_business_date = latest_business_date

        if min_business_date >= max_business_date:
            # Fallback if the constraints don't work - create a recent business
            creation_date = sim_start_date - \
                datetime.timedelta(days=self.random_instance.randint(30, 365))
        else:
            time_delta = (max_business_date -
                          min_business_date).total_seconds()
            creation_date = min_business_date + \
                datetime.timedelta(
                    seconds=self.random_instance.randint(0, int(time_delta)))

        business_category = business_category_override if business_category_override else generate_business_category(
            self.faker)

        # Default to owner's country, but calculate probability of being offshore
        country_code = owner_country if owner_country else self.random_instance.choice(
            COUNTRY_CODES)
        if owner_country and owner_occupation and owner_risk_score is not None:
            offshore_prob = self._calculate_offshore_probability(
                owner_occupation, owner_risk_score, business_category, owner_country)

            if self.random_instance.random() < offshore_prob:
                owner_country = country_code
                # 75% chance to be in a tax haven if going offshore
                if self.random_instance.random() < 0.75:
                    tax_havens = [
                        c for c in HIGH_RISK_COUNTRIES if c != owner_country]
                    if tax_havens:
                        country_code = self.random_instance.choice(tax_havens)
                else:
                    # Otherwise pick a non-tax haven foreign country
                    other_countries = [c for c in COUNTRY_CODES if c !=
                                       owner_country and c not in HIGH_RISK_COUNTRIES]
                    if other_countries:
                        country_code = self.random_instance.choice(
                            other_countries)

        is_in_high_risk_category = business_category in HIGH_RISK_BUSINESS_CATEGORIES
        is_in_high_risk_country = country_code in HIGH_RISK_COUNTRIES

        company_size_range = self.params.get("company_size_range", [1, 1000])

        common_attrs = {
            "country_code": country_code,
            "is_fraudulent": False
        }
        specific_attrs = {
            "name": self.faker.company(),
            "business_category": business_category,
            "incorporation_year": creation_date.year,
            "number_of_employees": self.random_instance.randint(company_size_range[0], company_size_range[1]),
            "is_high_risk_category": is_in_high_risk_category,
            "is_high_risk_country": is_in_high_risk_country,
        }
        risk_score = self._calculate_risk_score(specific_attrs, common_attrs)
        common_attrs["risk_score"] = risk_score

        return (creation_date, common_attrs, specific_attrs)

    def to_node_attributes(self, common_attrs: Dict[str, Any], specific_attrs: Dict[str, Any]) -> NodeAttributes:
        """Convert business attributes to node attributes."""
        return super().to_node_attributes(common_attrs, specific_attrs)
