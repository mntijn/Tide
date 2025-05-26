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


class AccountCategory(Enum):
    CURRENT = "current"
    SAVINGS = "savings"
    LOAN = "loan"
    BUSINESS = "business"
    CASH = "cash"


class AgeGroup(Enum):
    EIGHTEEN_TO_TWENTY_FOUR = "18-24"
    TWENTY_FIVE_TO_THIRTY_FOUR = "25-34"
    THIRTY_FIVE_TO_FORTY_NINE = "35-49"
    FIFTY_TO_SIXTY_FOUR = "50-64"
    SIXTY_FIVE_PLUS = "65+"


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"


@dataclass
class NodeAttributes:
    node_type: NodeType
    creation_date: Optional[datetime.datetime] = None
    address: Optional[Dict[str, str]] = None
    is_fraudulent: bool = False
    risk_score: Optional[float] = None


@dataclass
class IndividualAttributes:
    name: Optional[str] = None
    age_group: Optional[AgeGroup] = None
    occupation: Optional[str] = None
    gender: Optional[Gender] = None


@dataclass
class BusinessAttributes:
    name: Optional[str] = None
    business_category: Optional[str] = None
    incorporation_year: Optional[int] = None
    number_of_employees: Optional[int] = None
    is_high_risk_category: bool = False
    is_high_risk_country: bool = False


@dataclass
class InstitutionAttributes:
    name: Optional[str] = None


@dataclass
class AccountAttributes:
    start_balance: float = 0.0
    current_balance: float = 0.0
    institution_id: Optional[str] = None
    account_category: Optional[AccountCategory] = None
    currency: Optional[str] = None


@dataclass
class EdgeAttributes:
    edge_type: EdgeType


@dataclass
class TransactionAttributes(EdgeAttributes):
    timestamp: datetime.datetime = field(kw_only=True)
    amount: float = field(kw_only=True)
    currency: str = "EUR"
    transaction_type: TransactionType = field(kw_only=True)
    is_fraudulent: bool = False
    edge_type: EdgeType = EdgeType.TRANSACTION


@dataclass
class OwnershipAttributes(EdgeAttributes):
    ownership_start_date: datetime.date = field(kw_only=True)
    ownership_percentage: Optional[float] = None
    edge_type: EdgeType = EdgeType.OWNERSHIP
