from typing import Dict, Any, Tuple, List, Optional
import datetime
from ..datastructures.enums import NodeType, AgeGroup
from ..datastructures.attributes import NodeAttributes
from ..utils.constants import (
    COUNTRY_TO_CURRENCY,
    COUNTRY_CODES,
    HIGH_RISK_COUNTRIES,
    HIGH_RISK_OCCUPATIONS,
    HIGH_PAID_OCCUPATIONS
)
from .base import Entity


class Account(Entity):
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

        # Get balance ranges with fallbacks
        self.balance_range_normal = params.get(
            "account_balance_range_normal", [1000, 50000])
        self.balance_range_offshore = params.get(
            "account_balance_range_offshore", self.balance_range_normal)

    def _calculate_offshore_probability(self, entity_node_type: NodeType, entity_data: Dict[str, Any]) -> float:
        """Calculate the probability of an account being offshore based on entity attributes."""
        probability = self.base_offshore_probability

        if entity_node_type == NodeType.INDIVIDUAL:
            risk_score = entity_data.get("risk_score", 0.0)
            probability += risk_score * 0.05

            # Increase probability for business owners/entrepreneurs and high-paid occupations
            occupation = entity_data.get("occupation", "")
            if occupation in HIGH_RISK_OCCUPATIONS:  # These are often business-related
                probability += 0.02
            if occupation in HIGH_PAID_OCCUPATIONS:  # Higher chance for high-paid individuals
                probability += 0.08

            # Increase probability for older individuals (more likely to have accumulated wealth)
            age_group = entity_data.get("age_group")
            if age_group in [AgeGroup.FIFTY_TO_SIXTY_FOUR, AgeGroup.SIXTY_FIVE_PLUS]:
                probability += 0.03

            # Increase probability for people from high-risk countries
            country_code = entity_data.get("country_code")
            if country_code in HIGH_RISK_COUNTRIES:
                probability += 0.02

        elif entity_node_type == NodeType.BUSINESS:
            # Businesses are more likely to have offshore accounts
            probability += 0.05

            # Increase probability for businesses in high-risk countries
            country_code = entity_data.get("country_code")
            if country_code in HIGH_RISK_COUNTRIES:
                probability += 0.04

        # Cap the probability at 0.25 (25%)
        return min(probability, 0.25)

    def generate_accounts_and_ownership_data_for_entity(
        self,
        entity_node_type: NodeType,
        entity_country_code: str,
        entity_data: Dict[str, Any],
        sim_start_date: datetime.datetime,
        entity_creation_date: Optional[datetime.datetime] = None,
    ) -> List[Tuple[datetime.datetime, Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
        """Generates data for account nodes and their ownership edges for a single entity."""
        accounts_and_ownerships_data = []

        if not self.all_institution_ids:
            return []

        if not entity_creation_date:
            entity_creation_date = sim_start_date

        num_accounts_range_key = ""
        if entity_node_type == NodeType.INDIVIDUAL:
            num_accounts_range_key = "individual_accounts_per_institution_range"
        elif entity_node_type == NodeType.BUSINESS:
            num_accounts_range_key = "business_accounts_per_institution_range"
        else:
            return []

        min_acc, max_acc = self.graph_scale.get(num_accounts_range_key, (1, 1))
        num_accounts_to_create = self.random_instance.randint(min_acc, max_acc)

        # Calculate offshore probability based on entity attributes
        offshore_probability = self._calculate_offshore_probability(
            entity_node_type, entity_data)

        for _ in range(num_accounts_to_create):
            min_date = entity_creation_date
            max_date = sim_start_date
            time_delta_days = (max_date - min_date).days
            acc_creation_offset_days = self.random_instance.randint(
                0, time_delta_days)
            acc_creation_date = min_date + \
                datetime.timedelta(days=acc_creation_offset_days)

            account_country_code = entity_country_code

            # Adjust balance ranges for high-paid occupations
            if entity_node_type == NodeType.INDIVIDUAL and entity_data.get("occupation") in HIGH_PAID_OCCUPATIONS:
                # High-paid occupations get 3x higher balance ranges
                balance_range = [x * 3 for x in self.balance_range_normal]
            else:
                balance_range = self.balance_range_normal

            # Decide if this is an offshore account using the calculated probability
            if self.random_instance.random() < offshore_probability:
                # 14% chance to be in a tax haven
                if self.random_instance.random() < 0.14:
                    # Try to find a tax haven country
                    tax_havens = [
                        c for c in HIGH_RISK_COUNTRIES if c != entity_country_code]
                    if tax_havens:
                        account_country_code = self.random_instance.choice(
                            tax_havens)
                        # For high-paid occupations, increase tax haven balances even more
                        if entity_node_type == NodeType.INDIVIDUAL and entity_data.get("occupation") in HIGH_PAID_OCCUPATIONS:
                            # 8x for high-paid in tax havens
                            balance_range = [
                                x * 8 for x in self.balance_range_normal]
                        else:
                            balance_range = self.balance_range_offshore
                else:
                    # Try to find a non-tax haven country
                    other_countries = [c for c in COUNTRY_CODES if c !=
                                       entity_country_code and c not in HIGH_RISK_COUNTRIES]
                    if other_countries:
                        account_country_code = self.random_instance.choice(
                            other_countries)
                        # For high-paid occupations, increase offshore balances
                        if entity_node_type == NodeType.INDIVIDUAL and entity_data.get("occupation") in HIGH_PAID_OCCUPATIONS:
                            # 5x for high-paid offshore
                            balance_range = [
                                x * 5 for x in self.balance_range_normal]
                        else:
                            balance_range = self.balance_range_offshore

            # Choose an institution from the account's country
            institutions_in_country = [
                inst_id for inst_id in self.all_institution_ids
                if self.institution_countries.get(inst_id) == account_country_code
            ]

            # Fallback to any institution if none found in the target country
            if institutions_in_country:
                chosen_institution_id = self.random_instance.choice(
                    institutions_in_country)
            else:
                chosen_institution_id = self.random_instance.choice(
                    self.all_institution_ids)

            start_balance = round(self.random_instance.uniform(
                balance_range[0], balance_range[1]), 2)
            currency = self.currency_mapping.get(account_country_code)

            acc_common_attrs = {
                "country_code": account_country_code,
                "is_fraudulent": False
            }
            acc_specific_attrs = {
                "start_balance": start_balance,
                "current_balance": start_balance,
                "institution_id": chosen_institution_id,
                "account_category": self.random_instance.choice(self.account_categories),
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

    def to_node_attributes(self, common_attrs: Dict[str, Any], specific_attrs: Dict[str, Any]) -> NodeAttributes:
        """Convert account attributes to node attributes."""
        return super().to_node_attributes(common_attrs, specific_attrs)
