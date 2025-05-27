from typing import Dict, Any, Tuple, List
import datetime
import random
from faker import Faker
from ..datastructures.enums import NodeType, AgeGroup
from ..datastructures.attributes import NodeAttributes
from ..utils.constants import COUNTRY_TO_CURRENCY, COUNTRY_CODES, HIGH_RISK_COUNTRIES, HIGH_RISK_OCCUPATIONS
from ..utils.address import generate_localized_address
from .base import BaseEntity


class Account(BaseEntity):
    def __init__(self, params: Dict[str, Any], all_institution_ids: List[str], institution_countries: Dict[str, str]):
        super().__init__(params)
        if not all_institution_ids:
            print("Warning: Account initialized with no institution IDs.")
        self.all_institution_ids = all_institution_ids
        self.institution_countries = institution_countries
        self.account_balance_range = params.get("account_balance_range_normal")
        self.currency_mapping = COUNTRY_TO_CURRENCY
        self.account_categories = params.get("account_categories")
        self.base_offshore_probability = params.get(
            "offshore_account_probability", 0.01)

    def _calculate_offshore_probability(self, entity_node_type: NodeType, entity_data: Dict[str, Any]) -> float:
        """Calculate the probability of an account being offshore based on entity attributes."""
        probability = self.base_offshore_probability

        if entity_node_type == NodeType.INDIVIDUAL:
            # Increase probability based on risk score (wealthier/more sophisticated individuals)
            risk_score = entity_data.get("risk_score", 0.0)
            probability += risk_score * 0.05

            # Increase probability for business owners/entrepreneurs
            occupation = entity_data.get("occupation", "")
            if occupation in HIGH_RISK_OCCUPATIONS:  # These are often business-related
                probability += 0.03

            # Increase probability for older individuals (more likely to have accumulated wealth)
            age_group = entity_data.get("age_group")
            if age_group in [AgeGroup.FIFTY_TO_SIXTY_FOUR, AgeGroup.SIXTY_FIVE_PLUS]:
                probability += 0.02

            # Increase probability for people from high-risk countries
            country_code = entity_data.get("address", {}).get("country")
            if country_code in HIGH_RISK_COUNTRIES:
                probability += 0.04

        elif entity_node_type == NodeType.BUSINESS:
            # Businesses are more likely to have offshore accounts, but still not guaranteed
            probability += 0.05

            # Increase probability for businesses in high-risk countries
            country_code = entity_data.get("address", {}).get("country")
            if country_code in HIGH_RISK_COUNTRIES:
                probability += 0.04

        # Cap the probability at 0.15 (15%) - reduced from 0.8
        return min(probability, 0.15)

    def generate_accounts_and_ownership_data_for_entity(
        self,
        entity_node_type: NodeType,
        entity_creation_date: datetime.datetime,
        entity_country_code: str,
        entity_address: Dict[str, Any],
        entity_data: Dict[str, Any],  # Added entity_data parameter
        sim_start_date: datetime.datetime
    ) -> List[Tuple[datetime.datetime, Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
        """Generates data for account nodes and their ownership edges for a single entity."""
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

        # Calculate offshore probability based on entity attributes
        offshore_probability = self._calculate_offshore_probability(
            entity_node_type, entity_data)

        for _ in range(num_accounts_to_create):
            min_date = entity_creation_date
            max_date = sim_start_date
            time_delta_seconds = (max_date - min_date).total_seconds()
            acc_creation_offset_seconds = random.randint(
                0, int(time_delta_seconds))
            acc_creation_date = min_date + \
                datetime.timedelta(seconds=acc_creation_offset_seconds)

            chosen_institution_id = random.choice(self.all_institution_ids)
            start_balance = round(random.uniform(
                self.account_balance_range[0], self.account_balance_range[1]), 2)

            account_country_code = entity_country_code
            account_address = entity_address

            # Decide if this is an offshore account using the calculated probability
            if random.random() < offshore_probability:
                # Select an offshore country different from the entity's country
                possible_offshore_countries = [
                    c for c in COUNTRY_CODES if c != entity_country_code]
                if possible_offshore_countries:  # Ensure there's at least one other country
                    account_country_code = random.choice(
                        possible_offshore_countries)
                    account_address = generate_localized_address(
                        account_country_code)
                # If no other country, defaults to entity's country (edge case)

            institution_country_for_currency = self.institution_countries.get(
                chosen_institution_id)
            # For both offshore and non-offshore accounts, currency should match the account's country
            currency = self.currency_mapping.get(account_country_code)

            acc_common_attrs = {
                "address": account_address,
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
            print(
                f"creating account for {entity_node_type} {entity_data} in country {account_country_code} with currency {currency}")
            accounts_and_ownerships_data.append(
                (acc_creation_date, acc_common_attrs,
                 acc_specific_attrs, ownership_specific_attrs)
            )
        return accounts_and_ownerships_data

    def to_node_attributes(self, common_attrs: Dict[str, Any], specific_attrs: Dict[str, Any]) -> NodeAttributes:
        """Convert account attributes to node attributes."""
        return super().to_node_attributes(common_attrs, specific_attrs)
