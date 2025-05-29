# TODO: maybe give the pattern manager awareness of previously
# selected fraudulent entities so it can bias its selection logic.

import random
from typing import Dict, Any, List

from .repeated_overseas_transfers import RepeatedOverseasTransfersPattern
from .rapid_fund_movement import RapidFundMovementPattern
from .front_business_activity import FrontBusinessPattern


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

    def get_available_patterns(self) -> List[str]:
        """Return list of available pattern names"""
        available = list(self.patterns.keys())
        return available
