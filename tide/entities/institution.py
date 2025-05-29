from typing import Dict, Any, Tuple, List
from ..utils.random_instance import random_instance
from ..datastructures.attributes import NodeAttributes
from ..utils.constants import COUNTRY_CODES
from .base import Entity


class Institution(Entity):
    def generate_data(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Generates data for institution nodes based on institutions per country."""
        institutions_data = []
        institutions_per_country = self.graph_scale.get(
            "institutions_per_country", 1)

        # Generate the specified number of institutions for each country
        for country_code in COUNTRY_CODES:
            for i in range(institutions_per_country):
                common_attrs = {
                    "country_code": country_code,
                    "is_fraudulent": False
                }
                specific_attrs = {
                    "name": f"Bank_{country_code}_{i+1}_{self.faker.company_suffix()}",
                }
                institutions_data.append((common_attrs, specific_attrs))

        return institutions_data

    def to_node_attributes(self, common_attrs: Dict[str, Any], specific_attrs: Dict[str, Any]) -> NodeAttributes:
        """Convert institution attributes to node attributes."""
        return super().to_node_attributes(common_attrs, specific_attrs)
