# TODO: maybe give the pattern manager awareness of previously
# selected fraudulent entities so it can bias its selection logic.

# TODO: add a pattern selector that can be used to select patterns based on the graph


import random
from typing import Dict, Any, List, Tuple

from ..datastructures.enums import NodeType
from ..datastructures.attributes import TransactionAttributes
from .repeated_overseas_transfers import RepeatedOverseasTransfersPattern
from .rapid_fund_movement import RapidFundMovementPattern
from .front_business_activity import FrontBusinessPattern
from .base import EntitySelection


class PatternManager:
    """Manages injection of all patterns"""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        self.graph_generator = graph_generator
        self.params = params

        self.patterns: Dict[str, Any] = {
            p.pattern_name: p for p in [
                RepeatedOverseasTransfersPattern(graph_generator, params),
                RapidFundMovementPattern(graph_generator, params),
                FrontBusinessPattern(graph_generator, params),
                # add new patterns here
            ]}

        # Initialize entity selector (set by graph generator)
        # With Option 2, GraphGenerator *is* the selector. PatternManager just holds patterns.
        # self.entity_selector = graph_generator # Deprecated: will be removed

    def get_available_patterns(self) -> List[str]:
        """Return list of available pattern names"""
        available = list(self.patterns.keys())
        return available
