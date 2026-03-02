from dataclasses import dataclass, field
import datetime

from .enums import (
    NodeType,
    EdgeType,
    TransactionType,
    AccountCategory,
    AgeGroup,
    Gender
)


@dataclass
class NodeAttributes:
    node_type: NodeType
    creation_date: datetime.datetime | None = None
    country_code: str | None = None
    is_fraudulent: bool = False
    risk_score: float | None = None


@dataclass
class IndividualAttributes:
    name: str | None = None
    age_group: AgeGroup | None = None
    occupation: str | None = None
    gender: Gender | None = None


@dataclass
class BusinessAttributes:
    name: str | None = None
    business_category: str | None = None
    incorporation_year: int | None = None
    number_of_employees: int | None = None
    is_high_risk_category: bool = False
    is_high_risk_country: bool = False


@dataclass
class InstitutionAttributes:
    name: str | None = None


@dataclass
class AccountAttributes:
    institution_id: str | None = None
    account_category: AccountCategory | None = None
    currency: str | None = None


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
    time_since_previous_transaction: datetime.timedelta | None = None


@dataclass
class OwnershipAttributes(EdgeAttributes):
    ownership_start_date: datetime.date = field(kw_only=True)
    ownership_percentage: float | None = None
    edge_type: EdgeType = EdgeType.OWNERSHIP
