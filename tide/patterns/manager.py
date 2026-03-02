from .repeated_overseas_transfers import RepeatedOverseasTransfersPattern
from .rapid_fund_movement import RapidFundMovementPattern
from .front_business_activity import FrontBusinessPattern
from .synchronised_transactions import SynchronisedTransactionsPattern
from .u_turn_transactions import UTurnTransactionsPattern
from typing import Any


class PatternManager:
    """Manages injection of all patterns"""

    def __init__(self, graph_generator, params: dict[str, Any]):
        self.graph_generator = graph_generator
        self.params = params

        self.patterns: dict[str, Any] = {
            p.pattern_name: p for p in [
                RepeatedOverseasTransfersPattern(graph_generator, params),
                RapidFundMovementPattern(graph_generator, params),
                FrontBusinessPattern(graph_generator, params),
                # SynchronisedTransactionsPattern(graph_generator, params),
                UTurnTransactionsPattern(graph_generator, params),
                # add new patterns here
            ]}

    def get_available_patterns(self) -> list[str]:
        """Return list of available pattern names (sorted for determinism)"""
        available = sorted(self.patterns.keys())
        return available
