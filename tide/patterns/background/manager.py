from typing import Dict, Any, List

from .background_activity import RandomPaymentsPattern
from .salary_payments import SalaryPaymentsPattern
from .fraudster_background import FraudsterBackgroundPattern
from .legitimate_high_payments import LegitimateHighPaymentsPattern
from .legitimate_burst import LegitimateBurstPattern
from .legitimate_cash_operations import LegitimateCashOperationsPattern
from .legitimate_periodic import LegitimatePeriodicPaymentsPattern
from .legitimate_chains import LegitimateChainsPattern
from .legitimate_structuring import LegitimateStructuringPattern
from .legitimate_rapid_flow import LegitimateRapidFlowPattern
from .legitimate_high_risk_activity import LegitimateHighRiskActivityPattern


class BackgroundPatternManager:
    """Manages injection of background patterns"""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        self.graph_generator = graph_generator
        self.params = params

        self.patterns: Dict[str, Any] = {
            p.pattern_name: p for p in [
                RandomPaymentsPattern(graph_generator, params),
                SalaryPaymentsPattern(graph_generator, params),
                FraudsterBackgroundPattern(graph_generator, params),
                LegitimateHighPaymentsPattern(graph_generator, params),
                LegitimateBurstPattern(graph_generator, params),
                LegitimateCashOperationsPattern(graph_generator, params),
                LegitimatePeriodicPaymentsPattern(graph_generator, params),
                LegitimateChainsPattern(graph_generator, params),
                # Patterns to reduce feature-based data leakage
                LegitimateStructuringPattern(graph_generator, params),
                LegitimateRapidFlowPattern(graph_generator, params),
                # Breaks risk_score -> fraud correlation
                LegitimateHighRiskActivityPattern(graph_generator, params),
            ]
        }

    def get_available_patterns(self) -> List[str]:
        """Return list of available background pattern names (sorted for determinism)"""
        available = sorted(self.patterns.keys())
        return available
