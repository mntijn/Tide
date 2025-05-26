"""Patterns module for TIDE"""

from .base import (
    PatternInjector,
    StructuralComponent,
    TemporalComponent,
    CompositePattern,
    EntitySelection,
    TransactionSequence
)

# Import specific pattern implementations
from .repeated_overseas_transfers import RepeatedOverseasTransfersPattern
from .rapid_fund_movement import RapidFundMovementPattern
from .front_business_activity import FrontBusinessPattern

# Import the manager
from .manager import PatternManager

__all__ = [
    "PatternInjector",
    "StructuralComponent",
    "TemporalComponent",
    "CompositePattern",
    "EntitySelection",
    "TransactionSequence",
    "HighFrequencyTransactionPattern",
    "SynchronizedTransactionPattern",
    "UTurnTransactionPattern",
    "StarTopologyPattern",
    "RepeatedOverseasTransfersPattern",
    "RapidFundMovementPattern",
    "FrontBusinessPattern",
    "PatternManager"
]
