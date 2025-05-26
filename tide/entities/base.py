from typing import Dict, Any
from faker import Faker
from ..datastructures.attributes import NodeAttributes
from ..utils.constants import HIGH_RISK_COUNTRIES


class BaseEntity:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.faker = Faker()
        self.graph_scale = params.get("graph_scale", {})
        self.time_span = params.get("time_span", {})
        self.risk_config = self.params.get("high_risk_config", {})
        self.risk_weights = self.params.get("risk_weights", {})

    def _calculate_base_risk_score(self, common_attrs: Dict[str, Any]) -> float:
        """Base risk score calculation that can be extended by specific entities."""
        base_risk = self.risk_weights.get("base_risk", 0.05)
        score = base_risk

        entity_country = common_attrs.get("address", {}).get("country")
        if entity_country in HIGH_RISK_COUNTRIES:
            country_weight = self.risk_weights.get("country", 0.20)
            country_factor = self.risk_config.get(
                "countries_weight_factor", 1.0)
            score += country_weight * country_factor

        max_score = self.risk_weights.get("max_score", 0.9)
        return min(score, max_score)

    def to_node_attributes(self, common_attrs: Dict[str, Any], specific_attrs: Dict[str, Any]) -> NodeAttributes:
        """Convert entity attributes to node attributes."""
        return NodeAttributes(
            common_attrs=common_attrs,
            specific_attrs=specific_attrs
        )
