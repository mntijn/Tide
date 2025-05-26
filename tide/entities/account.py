from typing import Dict, Any, Tuple, List
import datetime
import random
from faker import Faker
from ..datastructures.enums import NodeType
from ..datastructures.attributes import NodeAttributes
from ..utils.constants import COUNTRY_TO_CURRENCY
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

    def generate_accounts_and_ownership_data_for_entity(
        self,
        entity_node_type: NodeType,
        entity_creation_date: datetime.datetime,
        entity_country_code: str,
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
            currency = self.currency_mapping.get(institution_country)

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

    def to_node_attributes(self, common_attrs: Dict[str, Any], specific_attrs: Dict[str, Any]) -> NodeAttributes:
        """Convert account attributes to node attributes."""
        return super().to_node_attributes(common_attrs, specific_attrs)
