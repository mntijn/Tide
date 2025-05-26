from enum import Enum


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
