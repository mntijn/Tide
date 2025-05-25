import random
import datetime
from faker import Faker
from typing import List, Dict, Any, Tuple, Optional
import random

from .data_structures import (
    NodeType, EdgeType, NodeAttributes, AccountAttributes,
    IndividualAttributes, BusinessAttributes, InstitutionAttributes,
    OwnershipAttributes, AgeGroup, Gender
)
from .utils import COUNTRY_CODES, generate_localized_address, HIGH_RISK_BUSINESS_CATEGORIES, HIGH_RISK_COUNTRIES, HIGH_RISK_OCCUPATIONS, HIGH_RISK_AGE_GROUPS, generate_business_category, COUNTRY_TO_CURRENCY


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


class BaseCreator:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.faker = Faker()
        self.graph_scale = params.get("graph_scale", {})
        self.time_span = params.get("time_span", {})
        self.risk_config = self.params.get("high_risk_config", {})
        self.risk_weights = self.params.get("risk_weights", {})


class InstitutionCreator(BaseCreator):
    def generate_institutions_data(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Generates data for institution nodes.
        """
        institutions_data = []
        num_institutions = self.graph_scale.get("institutions", 0)

        for i in range(num_institutions):
            country_code = random.choice(COUNTRY_CODES)
            common_attrs = {
                "address": generate_localized_address(country_code),
                "is_fraudulent": False
            }
            specific_attrs = {
                "name": f"Bank_{i+1}_{self.faker.company_suffix()}",
            }
            institutions_data.append((common_attrs, specific_attrs))
        return institutions_data


class IndividualCreator(BaseCreator):
    def _calculate_risk_score(self, specific_attrs: Dict[str, Any], common_attrs: Dict[str, Any]) -> float:
        """Calculates a risk score for an individual."""
        base_risk = self.risk_weights.get("base_individual", 0.05)
        score = base_risk

        # Use high-risk age groups from utils.py instead of config
        if specific_attrs.get("age_group").name in HIGH_RISK_AGE_GROUPS:
            score += self.risk_weights.get("age_group", 0.15)

        if specific_attrs.get("occupation") in HIGH_RISK_OCCUPATIONS:
            score += self.risk_weights.get("occupation", 0.10)

        entity_country = common_attrs.get("address", {}).get("country")
        if entity_country in HIGH_RISK_COUNTRIES:
            country_weight = self.risk_weights.get("country", 0.20)
            country_factor = self.risk_config.get(
                "countries_weight_factor", 1.0)
            score += country_weight * country_factor

        max_score = self.risk_weights.get("max_score", 0.9)
        return min(score, max_score)

    def generate_individuals_data(self) -> List[Tuple[datetime.datetime, Dict[str, Any], Dict[str, Any]]]:
        """
        Generates data for individual nodes, including a risk score.
        """
        individuals_data = []
        num_individuals = self.graph_scale.get("individuals", 0)
        start_date = self.time_span["start_date"]

        for _ in range(num_individuals):
            creation_date = start_date - \
                datetime.timedelta(days=random.randint(30, 1000))
            country_code = random.choice(COUNTRY_CODES)
            common_attrs = {
                "address": generate_localized_address(country_code),
                "is_fraudulent": False  # Default to False, will be updated later
            }
            specific_attrs = {
                "name": self.faker.name(),
                "age_group": random.choice(list(AgeGroup)),
                "occupation": self.faker.job(),
                "gender": random.choice(list(Gender)),
            }
            # Calculate and add risk score
            risk_score = self._calculate_risk_score(
                specific_attrs, common_attrs)
            common_attrs["risk_score"] = risk_score

            individuals_data.append(
                (creation_date, common_attrs, specific_attrs))
        return individuals_data


class BusinessCreator(BaseCreator):
    def _calculate_risk_score(self, specific_attrs: Dict[str, Any], common_attrs: Dict[str, Any]) -> float:
        """Calculates a risk score for a business."""
        base_risk = self.risk_weights.get("base_business", 0.1)
        score = base_risk

        business_category = specific_attrs.get("business_category")
        if business_category in HIGH_RISK_BUSINESS_CATEGORIES:
            category_weight = self.risk_weights.get("business_category", 0.25)
            category_factor = self.risk_config.get(
                "business_categories_weight_factor", 1.0)
            score += category_weight * category_factor

        entity_country = common_attrs.get("address", {}).get("country")
        if entity_country in HIGH_RISK_COUNTRIES:
            country_weight = self.risk_weights.get("country", 0.20)
            country_factor = self.risk_config.get(
                "countries_weight_factor", 1.0)
            score += country_weight * country_factor

        num_employees = specific_attrs.get("number_of_employees", 0)
        company_size_thresholds = self.risk_config.get(
            "company_size_thresholds", {})

        # Very small companies (1-5 employees) can be shell companies
        if num_employees <= company_size_thresholds.get("very_small_max", 5):
            score += self.risk_weights.get("very_small_company", 0.10)

        max_score = self.risk_weights.get("max_score", 0.9)
        return min(score, max_score)

    def generate_businesses_data(self) -> List[Tuple[datetime.datetime, Dict[str, Any], Dict[str, Any]]]:
        """
        Generates data for business nodes, including a risk score.
        """
        businesses_data = []
        num_businesses = self.graph_scale.get("businesses", 0)
        start_date = self.time_span["start_date"]

        company_size_range = self.params.get("company_size_range")
        business_creation_range = self.params.get(
            "business_creation_date_range", [90, 5475])

        for _ in range(num_businesses):
            creation_date = start_date - \
                datetime.timedelta(days=random.randint(
                    business_creation_range[0], business_creation_range[1]))
            country_code = random.choice(COUNTRY_CODES)
            business_category = generate_business_category(self.faker)

            is_in_high_risk_category = business_category in HIGH_RISK_BUSINESS_CATEGORIES
            is_in_high_risk_country = country_code in HIGH_RISK_COUNTRIES

            common_attrs = {
                "address": generate_localized_address(country_code),
                "is_fraudulent": False  # Default to False
            }
            specific_attrs = {
                "name": self.faker.company(),
                "business_category": business_category,
                "incorporation_year": creation_date.year,
                "number_of_employees": random.randint(company_size_range[0], company_size_range[1]),
                "is_high_risk_category": is_in_high_risk_category,
                "is_high_risk_country": is_in_high_risk_country,
            }
            # Calculate and add risk score
            risk_score = self._calculate_risk_score(
                specific_attrs, common_attrs)
            common_attrs["risk_score"] = risk_score

            businesses_data.append(
                (creation_date, common_attrs, specific_attrs))
        return businesses_data

    def generate_age_consistent_business_for_individual(
        self,
        individual_age_group: AgeGroup,
        individual_creation_date: datetime.datetime,
        sim_start_date: datetime.datetime
    ) -> Tuple[datetime.datetime, Dict[str, Any], Dict[str, Any]]:
        """
        Generates a business that is consistent with the given individual's age, including a risk score.
        """
        individual_max_age = get_max_age_from_group(individual_age_group)

        # Business can't be older than when individual was 18
        # Assume individual was 18 at their creation_date - (individual_max_age - 18) years
        individual_18th_birthday = individual_creation_date - \
            datetime.timedelta(days=(individual_max_age - 18) * 365)

        # Business creation date must be after individual turned 18, but before sim_start_date
        min_business_date = max(individual_18th_birthday, sim_start_date - datetime.timedelta(
            days=self.params.get("business_creation_date_range", [90, 5475])[1]))
        max_business_date = sim_start_date - \
            datetime.timedelta(days=self.params.get(
                "business_creation_date_range", [90, 5475])[0])

        # Ensure we have a valid date range
        if min_business_date >= max_business_date:
            creation_date = sim_start_date - \
                datetime.timedelta(days=random.randint(30, 365))
        else:
            time_delta = (max_business_date -
                          min_business_date).total_seconds()
            creation_date = min_business_date + \
                datetime.timedelta(seconds=random.randint(0, int(time_delta)))

        country_code = random.choice(COUNTRY_CODES)
        business_category = generate_business_category(self.faker)

        is_in_high_risk_category = business_category in HIGH_RISK_BUSINESS_CATEGORIES
        is_in_high_risk_country = country_code in HIGH_RISK_COUNTRIES

        company_size_range = self.params.get("company_size_range", [1, 1000])

        common_attrs = {
            "address": generate_localized_address(country_code),
            "is_fraudulent": False  # Default to False
        }
        specific_attrs = {
            "name": self.faker.company(),
            "business_category": business_category,
            "incorporation_year": creation_date.year,
            "number_of_employees": random.randint(company_size_range[0], company_size_range[1]),
            "is_high_risk_category": is_in_high_risk_category,
            "is_high_risk_country": is_in_high_risk_country,
        }
        # Calculate and add risk score
        risk_score = self._calculate_risk_score(specific_attrs, common_attrs)
        common_attrs["risk_score"] = risk_score

        return (creation_date, common_attrs, specific_attrs)


class AccountCreator(BaseCreator):
    def __init__(self, params: Dict[str, Any], all_institution_ids: List[str], institution_countries: Dict[str, str]):
        super().__init__(params)
        if not all_institution_ids:
            print("Warning: AccountCreator initialized with no institution IDs.")
        self.all_institution_ids = all_institution_ids

        self.institution_countries = institution_countries
        self.account_balance_range = params.get("account_balance_range_normal")
        self.currency_mapping = COUNTRY_TO_CURRENCY
        self.account_categories = params.get("account_categories")

    def generate_accounts_and_ownership_data_for_entity(
        self,
        entity_node_type: NodeType,
        entity_creation_date: datetime.datetime,
        entity_country_code: str,
        sim_start_date: datetime.datetime
    ) -> List[Tuple[datetime.datetime, Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
        """
        Generates data for account nodes and their ownership edges for a single entity.
        """
        accounts_and_ownerships_data = []

        if not self.all_institution_ids:
            return []

        num_accounts_range_key = ""
        if entity_node_type == NodeType.INDIVIDUAL:
            num_accounts_range_key = "individual_accounts_per_institution_range"
        elif entity_node_type == NodeType.BUSINESS:
            num_accounts_range_key = "business_accounts_per_institution_range"
        else:
            return []

        min_acc, max_acc = self.graph_scale.get(num_accounts_range_key, (1, 1))
        num_accounts_to_create = random.randint(min_acc, max_acc)

        for _ in range(num_accounts_to_create):
            min_date = entity_creation_date
            max_date = sim_start_date

            time_delta_seconds = (max_date - min_date).total_seconds()

            acc_creation_offset_seconds = random.randint(
                0, int(time_delta_seconds))
            acc_creation_date = min_date + \
                datetime.timedelta(seconds=acc_creation_offset_seconds)

            chosen_institution_id = random.choice(self.all_institution_ids)
            start_balance = random.uniform(
                self.account_balance_range[0], self.account_balance_range[1])

            institution_country = self.institution_countries.get(
                chosen_institution_id)
            currency = self.currency_mapping.get(
                institution_country)

            acc_common_attrs = {
                "address": generate_localized_address(entity_country_code),
                "is_fraudulent": False
            }
            acc_specific_attrs = {
                "start_balance": start_balance,
                "current_balance": start_balance,
                "institution_id": chosen_institution_id,
                "account_category": random.choice(self.account_categories),
                "currency": currency,
            }
            ownership_specific_attrs = {
                "ownership_start_date": acc_creation_date.date(),
                "ownership_percentage": 100.0,
            }
            accounts_and_ownerships_data.append(
                (acc_creation_date, acc_common_attrs,
                 acc_specific_attrs, ownership_specific_attrs)
            )
        return accounts_and_ownerships_data
