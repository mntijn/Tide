from typing import Dict, Any, Tuple, List
import random
from faker import Faker
from ..datastructures.attributes import NodeAttributes
from ..utils.constants import COUNTRY_CODES
from ..utils.address import generate_localized_address
from .base import BaseEntity


class Institution(BaseEntity):
    def generate_data(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Generates data for institution nodes."""
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

    def to_node_attributes(self, common_attrs: Dict[str, Any], specific_attrs: Dict[str, Any]) -> NodeAttributes:
        """Convert institution attributes to node attributes."""
        return super().to_node_attributes(common_attrs, specific_attrs)
