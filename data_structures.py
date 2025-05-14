# All possible node types and edge types, and their attributes

from enum import Enum
from dataclasses import dataclass, field
import datetime
from typing import List, Dict, Any, Optional, Tuple


class NodeType(Enum):
    ACCOUNT = "account"
    INDIVIDUAL = "individual"
    BUSINESS = "business"
    INSTITUTION = "institution"


class EdgeType(Enum):
    TRANSACTION = "transaction"
    OWNERSHIP = "ownership"


class TransactionType(Enum):
    PAYMENT = "payment"
    SALARY = "salary"
    TRANSFER = "transfer"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"


@dataclass
class NodeAttributes:
    node_type: NodeType
    # institutions dont have a creation date
    creation_date: Optional[datetime.datetime] = None
    geo_location: str = "UNKNOWN"
    is_fraudulent: bool = False


@dataclass
class AccountAttributes:
    start_balance: float = 0.0
    current_balance: float = 0.0
    institution_id: str = None


@dataclass
class EdgeAttributes:
    edge_type: EdgeType


@dataclass
class TransactionAttributes(EdgeAttributes):
    timestamp: datetime.datetime = field(kw_only=True)
    amount: float = field(kw_only=True)
    currency: str = "EUR"
    transaction_type: TransactionType = field(kw_only=True)
    edge_type: EdgeType = EdgeType.TRANSACTION


@dataclass
class OwnershipAttributes(EdgeAttributes):
    ownership_start_date: datetime.date = field(kw_only=True)
    ownership_percentage: Optional[float] = None
    edge_type: EdgeType = EdgeType.OWNERSHIP
