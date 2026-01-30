"""
Amount interleaving utilities for feature distribution matching.

This module provides functions to generate amounts that overlap between
fraud and legitimate patterns, preventing amount-based data leakage.

Key insight: Fraud patterns use `generate_structured_amounts` which produces
amounts at 70-95% of the reporting threshold. To prevent leakage, legitimate
patterns should sometimes use similar distributions, and fraud patterns
should sometimes use "normal" distributions.
"""
from typing import List, Optional, Tuple
from .random_instance import random_instance
from .currency_conversion import generate_structured_amounts, convert_currency
from .amount_distributions import sample_lognormal_scalar


def generate_interleaved_amounts(
    count: int,
    normal_range: Tuple[float, float],
    structured_probability: float = 0.3,
    reporting_threshold: float = 10000.0,
    reporting_currency: str = "USD",
    target_currency: str = None,
) -> List[float]:
    """
    Generate amounts that mix normal and structured distributions.

    This prevents amount-based leakage by making both fraud and legitimate
    patterns use overlapping amount distributions.

    Args:
        count: Number of amounts to generate
        normal_range: (min, max) for normal uniform distribution
        structured_probability: Probability of using structured amount (default 0.3)
        reporting_threshold: Threshold for structuring (default $10,000)
        reporting_currency: Currency for threshold (default USD)
        target_currency: Target currency for amounts (defaults to reporting_currency)

    Returns:
        List of amounts with mixed distribution
    """
    target_currency = target_currency or reporting_currency
    amounts = []

    for _ in range(count):
        if random_instance.random() < structured_probability:
            # Generate structured amount (mimics fraud pattern)
            structured = generate_structured_amounts(
                count=1,
                reporting_threshold=reporting_threshold,
                reporting_currency=reporting_currency,
                target_currency=target_currency,
            )[0]
            amounts.append(structured)
        else:
            # Generate normal amount from log-normal (matches legitimate)
            amount = sample_lognormal_scalar("payment")
            amounts.append(round(amount, 2))

    return amounts


def generate_fraud_like_amounts(
    count: int,
    use_normal_probability: float = 0.25,
    normal_range: Tuple[float, float] = (500.0, 5000.0),
    reporting_threshold: float = 10000.0,
    reporting_currency: str = "USD",
    target_currency: str = None,
) -> List[float]:
    """
    Generate amounts for fraud patterns that sometimes use normal distributions.

    This prevents fraud amounts from being too distinctive by occasionally
    using "normal looking" amounts instead of always structured ones.

    Args:
        count: Number of amounts to generate
        use_normal_probability: Probability of using normal distribution (default 0.25)
        normal_range: Range for normal amounts when used
        reporting_threshold: Threshold for structuring
        reporting_currency: Currency for threshold
        target_currency: Target currency for amounts

    Returns:
        List of amounts for fraud transactions
    """
    target_currency = target_currency or reporting_currency
    amounts = []

    for _ in range(count):
        if random_instance.random() < use_normal_probability:
            # Use log-normal distribution (matches legitimate background)
            amount = sample_lognormal_scalar("payment")
            amounts.append(round(amount, 2))
        else:
            # Use structured amount (typical fraud pattern)
            structured = generate_structured_amounts(
                count=1,
                reporting_threshold=reporting_threshold,
                reporting_currency=reporting_currency,
                target_currency=target_currency,
            )[0]
            amounts.append(structured)

    return amounts


def generate_legitimate_structuring_amounts(
    count: int,
    tier_probabilities: dict = None,
) -> List[float]:
    """
    Generate amounts that look like legitimate structuring.

    Legitimate reasons for amounts just below $10K:
    - House down payment savings
    - Small business daily deposits
    - Tax-conscious transfers
    - Rent payments in high-cost areas

    Args:
        count: Number of amounts to generate
        tier_probabilities: Override probabilities for each tier

    Returns:
        List of amounts in various ranges including structuring range
    """
    default_tiers = {
        "small": {"range": (100.0, 1000.0), "prob": 0.25},
        "medium": {"range": (1000.0, 5000.0), "prob": 0.30},
        "large": {"range": (5000.0, 7500.0), "prob": 0.20},
        # Overlaps with fraud
        "structuring": {"range": (7500.0, 9999.0), "prob": 0.25},
    }

    if tier_probabilities:
        for tier, config in tier_probabilities.items():
            if tier in default_tiers:
                default_tiers[tier].update(config)

    # Build probability distribution (sorted for deterministic order)
    tiers = sorted(default_tiers.keys())
    probs = [default_tiers[t]["prob"] for t in tiers]

    amounts = []
    for _ in range(count):
        tier = random_instance.choices(tiers, weights=probs)[0]

        if tier == "structuring":
            # Structuring tier: keep uniform near threshold
            tier_range = default_tiers[tier]["range"]
            amount = random_instance.uniform(tier_range[0], tier_range[1])
        elif tier in ("small", "medium"):
            # Use log-normal for realistic amounts
            amount = sample_lognormal_scalar("payment")
        else:
            # Large tier
            amount = sample_lognormal_scalar("transfer")

        # Round to realistic denominations
        if amount > 5000:
            amount = round(amount / 100) * 100
        elif amount > 1000:
            amount = round(amount / 50) * 50
        else:
            amount = round(amount / 10) * 10

        amounts.append(max(10.0, amount))

    return amounts


def add_amount_noise(
    amount: float,
    noise_range: Tuple[float, float] = (-0.15, 0.15),
    min_amount: float = 10.0,
) -> float:
    """
    Add noise to an amount to blur distribution boundaries.

    Args:
        amount: Original amount
        noise_range: (min, max) percentage noise to add
        min_amount: Minimum allowed amount

    Returns:
        Amount with noise added
    """
    noise_pct = random_instance.uniform(noise_range[0], noise_range[1])
    noisy_amount = amount * (1 + noise_pct)
    return max(min_amount, round(noisy_amount, 2))


# =============================================================================
# NEW: Functions to lower ROC-AUC by making fraud look like AVERAGE users
# =============================================================================

def generate_fraud_with_camouflage(
    count: int,
    camouflage_probability: float = 0.15,
    small_amount_range: Tuple[float, float] = (50.0, 500.0),
    medium_amount_range: Tuple[float, float] = (500.0, 2000.0),
    high_amount_range: Tuple[float, float] = (7000.0, 9500.0),
) -> List[float]:
    """
    Generate fraud amounts with minimal camouflage.

    LIGHT INTERLEAVING: 15% camouflage keeps fraud mostly distinctive.
    - Most fraud (85%) retains characteristic high amounts
    - Small camouflage adds some realism without hurting GNN learning

    Args:
        count: Number of amounts to generate
        camouflage_probability: Probability of generating a "normal" looking amount
        small_amount_range: Range for small camouflage amounts
        medium_amount_range: Range for medium amounts (transition zone)
        high_amount_range: Range for high fraud amounts

    Returns:
        List of amounts with ~15% looking like normal users
    """
    amounts = []

    for _ in range(count):
        roll = random_instance.random()

        if roll < camouflage_probability * 0.6:
            # Small amount - log-normal payment (matches legitimate)
            amount = sample_lognormal_scalar("payment")
            if amount > 50:
                amount = round(amount / 10) * 10
            amounts.append(round(amount, 2))

        elif roll < camouflage_probability:
            # Medium amount - log-normal transfer (matches legitimate)
            amount = sample_lognormal_scalar("transfer")
            amount = round(amount / 50) * 50
            amounts.append(round(amount, 2))

        else:
            # High amount - structuring below threshold
            # 70-95% of reporting threshold ($10K)
            amount = random_instance.uniform(
                high_amount_range[0], high_amount_range[1])
            amount = round(amount / 100) * 100
            amounts.append(round(amount, 2))

    return amounts


def generate_varied_fraud_sequence(
    total_amount: float,
    num_transactions: int,
    include_test_txs: bool = True,
) -> List[float]:
    """
    Generate a sequence of varied amounts that sum to approximately total_amount.

    Instead of uniform splits (detectable pattern), uses varied amounts including
    small "test" transactions that look like average user behavior.

    Args:
        total_amount: Target total to move
        num_transactions: Number of transactions
        include_test_txs: Whether to include small test transactions

    Returns:
        List of amounts summing to ~total_amount, shuffled order
    """
    if num_transactions <= 0:
        return []
    if num_transactions == 1:
        return [round(total_amount, 2)]

    amounts = []
    remaining = total_amount

    if include_test_txs and num_transactions >= 3:
        # Add 1-2 small "test" transactions (looks like average user)
        num_test = min(2, num_transactions // 3)
        for _ in range(num_test):
            # Small test amount: $20-$100
            test_amount = random_instance.uniform(20.0, 100.0)
            test_amount = round(test_amount / 5) * 5
            amounts.append(test_amount)
            remaining -= test_amount

        # Remaining transactions split the rest
        num_remaining = num_transactions - num_test
    else:
        num_remaining = num_transactions

    # Split remaining amount with variation
    for i in range(num_remaining - 1):
        equal_split = remaining / (num_remaining - i)
        # Vary between 0.6x and 1.4x of equal split
        amount = random_instance.uniform(equal_split * 0.6, equal_split * 1.4)
        amount = max(50.0, min(amount, remaining - 50))
        amount = round(amount / 10) * 10
        amounts.append(amount)
        remaining -= amount

    # Final transaction gets remainder
    amounts.append(round(max(50.0, remaining), 2))

    random_instance.shuffle(amounts)
    return amounts
