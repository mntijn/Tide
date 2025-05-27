from typing import Dict, Any, Tuple, List
import datetime
import random
from faker import Faker
from ..datastructures.enums import AgeGroup, Gender
from ..datastructures.attributes import NodeAttributes
from ..utils.constants import (
    HIGH_RISK_AGE_GROUPS, HIGH_RISK_OCCUPATIONS, HIGH_RISK_COUNTRIES,
    COUNTRY_CODES
)
from ..utils.address import generate_localized_address
from ..utils.individual import generate_age_consistent_occupation
from .base import Entity


class Individual(Entity):
    def _calculate_risk_score(self, specific_attrs: Dict[str, Any], common_attrs: Dict[str, Any]) -> float:
        """Calculates a risk score for an individual."""
        score = self._calculate_base_risk_score(common_attrs)

        if specific_attrs.get("age_group").name in HIGH_RISK_AGE_GROUPS:
            score += self.risk_weights.get("age_group", 0.15)

        if specific_attrs.get("occupation") in HIGH_RISK_OCCUPATIONS:
            score += self.risk_weights.get("occupation", 0.10)

        max_score = self.risk_weights.get("max_score", 0.9)
        return min(score, max_score)

    def generate_data(self) -> List[Tuple[datetime.datetime, Dict[str, Any], Dict[str, Any]]]:
        """Generates data for individual nodes, including a risk score."""
        individuals_data = []
        num_individuals = self.graph_scale.get("individuals", 0)
        start_date = self.time_span["start_date"]

        for _ in range(num_individuals):
            creation_date = start_date - \
                datetime.timedelta(days=random.randint(30, 1000))
            country_code = random.choice(COUNTRY_CODES)
            common_attrs = {
                "address": generate_localized_address(country_code),
                "is_fraudulent": False
            }
            age_group = random.choice(list(AgeGroup))
            specific_attrs = {
                "name": self.faker.name(),
                "age_group": age_group,
                "occupation": generate_age_consistent_occupation(self.faker, age_group),
                "gender": random.choice(list(Gender)),
            }
            risk_score = self._calculate_risk_score(
                specific_attrs, common_attrs)
            common_attrs["risk_score"] = risk_score

            individuals_data.append(
                (creation_date, common_attrs, specific_attrs))
        return individuals_data

    def to_node_attributes(self, common_attrs: Dict[str, Any], specific_attrs: Dict[str, Any]) -> NodeAttributes:
        """Convert individual attributes to node attributes."""
        return super().to_node_attributes(common_attrs, specific_attrs)
