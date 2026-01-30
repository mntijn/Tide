"""
Realistic transaction amount sampling using log-normal distributions.

Real-world transaction amounts follow a log-normal distribution with Pareto
(power-law) tails. This module provides calibrated sampling functions based on
authoritative sources:

- 2022 Federal Reserve Payments Study (8th Triennial, March 2025)
- FinCEN SAR Statistics (2014-2024)
- Nacha ACH network reports
- IBM AMLworld NeurIPS 2023 methodology

Key insight: the body of real transaction amounts is log-normal, while the
tail follows a Pareto/power-law distribution with exponent ~1.16. We
approximate this using a log-normal with parameters calibrated to match
observed median and mean values per payment type.

Log-normal parameterization:
    If X ~ LogNormal(mu, sigma), then:
        median(X) = exp(mu)
        mean(X)   = exp(mu + sigma^2/2)

    Given target median M and mean E:
        mu    = ln(M)
        sigma = sqrt(2 * ln(E/M))
"""

import numpy as np
from .random_instance import get_numpy_rng, random_instance


# ============================================================================
# Default distribution parameters calibrated to real-world data
# These can be overridden via the YAML config
# ============================================================================

# Payment types and their calibrated log-normal parameters
# Source: Fed Payments Study 2022, Nacha, Nilson Report
#
# Format: {type: {"mu": float, "sigma": float, "min": float, "max": float}}
#   mu, sigma: log-normal parameters
#   min, max: hard bounds (clipping, not truncation)
DEFAULT_DISTRIBUTIONS = {
    "payment": {
        # Consumer payments (card-like): median ~$45, mean ~$64
        # Card payments: $9.76T / 153.3B txs ≈ $64 average
        # Median is lower due to skew
        "mu": 3.8,       # exp(3.8) ≈ $45 median
        "sigma": 1.2,    # Gives mean ~$93, with heavy right tail
        "min": 1.0,
        "max": 50000.0,   # B2B payments can be large
    },
    "transfer": {
        # ACH/wire transfers: median ~$250, mean ~$2,642
        # ACH: $91.85T / 34.7B txs ≈ $2,642 average
        # High sigma captures the enormous range (P2P $5 to B2B $100K+)
        "mu": 5.5,       # exp(5.5) ≈ $245 median
        "sigma": 2.0,    # Heavy tail reaching into tens of thousands
        "min": 5.0,
        "max": 500000.0,  # Business wire transfers reach hundreds of thousands
    },
    "deposit": {
        # Cash deposits: median ~$200, mean ~$800
        # Mix of ATM deposits, business cash, payroll
        "mu": 5.3,       # exp(5.3) ≈ $200 median
        "sigma": 1.5,    # Tail into $5K-$10K range (business deposits)
        "min": 10.0,
        "max": 100000.0,  # Business deposits can be very large
    },
    "withdrawal": {
        # ATM/cash withdrawals: Fed 2021 avg $198, implies median ~$122
        # Constrained by ATM limits ($500-$1000 typically)
        "mu": 4.8,       # exp(4.8) ≈ $122 median
        "sigma": 0.9,    # Moderate tail (ATM limits cap it)
        "min": 20.0,
        "max": 5000.0,
    },
    "salary": {
        # Direct deposit/payroll: median ~$1,600, mean ~$1,887
        # Source: Fed data on direct deposit average
        # Bimodal in reality (part-time vs full-time) but log-normal is
        # a reasonable single-mode approximation
        "mu": 7.4,       # exp(7.4) ≈ $1,636 median
        "sigma": 0.6,    # Moderate spread (salaries are more concentrated)
        "min": 200.0,
        "max": 30000.0,
    },
    # Special distributions for specific scenarios
    "high_value": {
        # Large transactions: business deals, property, etc.
        # median ~$15K, mean ~$50K
        "mu": 9.6,       # exp(9.6) ≈ $14,880 median
        "sigma": 1.5,    # Heavy tail into hundreds of thousands
        "min": 5000.0,
        "max": 1000000.0,
    },
    "cash_operations": {
        # General cash ops (deposit + withdrawal combined distribution)
        # median ~$120, mean ~$400
        "mu": 4.8,       # exp(4.8) ≈ $122 median
        "sigma": 1.3,
        "min": 10.0,
        "max": 15000.0,
    },
}


def sample_lognormal(
    tx_type: str,
    size: int = 1,
    config: dict = None,
    rng=None,
) -> np.ndarray:
    """
    Sample transaction amounts from a log-normal distribution.

    Args:
        tx_type: Transaction type key (e.g., "payment", "transfer", "salary")
        size: Number of samples
        config: Optional override dict with keys "mu", "sigma", "min", "max".
                Can also contain "distribution" key to select distribution type.
        rng: NumPy Generator instance. If None, uses the global seeded one.

    Returns:
        numpy array of sampled amounts, clipped to [min, max] and rounded.
    """
    if rng is None:
        rng = get_numpy_rng()

    # Get distribution parameters
    params = DEFAULT_DISTRIBUTIONS.get(tx_type, DEFAULT_DISTRIBUTIONS["payment"]).copy()

    # Apply config overrides if provided
    if config:
        if "mu" in config:
            params["mu"] = config["mu"]
        if "sigma" in config:
            params["sigma"] = config["sigma"]
        if "min" in config:
            params["min"] = config["min"]
        if "max" in config:
            params["max"] = config["max"]

    amounts = rng.lognormal(mean=params["mu"], sigma=params["sigma"], size=size)
    amounts = np.clip(amounts, params["min"], params["max"])
    amounts = np.round(amounts, 2)

    return amounts


def sample_lognormal_scalar(
    tx_type: str,
    config: dict = None,
) -> float:
    """
    Sample a single transaction amount from a log-normal distribution.

    Convenience wrapper around sample_lognormal for scalar use.
    """
    return float(sample_lognormal(tx_type, size=1, config=config)[0])


def sample_from_config(
    tx_type: str,
    size: int,
    amount_config: dict = None,
    rng=None,
) -> np.ndarray:
    """
    Sample amounts using config-driven distribution selection.

    The config can specify:
        distribution: "lognormal" (default) | "uniform"
        mu, sigma, min, max: for lognormal
        range: [lo, hi] for uniform (backward compatible)

    This is the primary entry point for all background patterns.
    """
    if rng is None:
        rng = get_numpy_rng()

    if amount_config is None:
        amount_config = {}

    dist_type = amount_config.get("distribution", "lognormal")

    if dist_type == "uniform":
        # Backward compatibility: uniform from range
        lo, hi = amount_config.get("range", [10.0, 500.0])
        return np.round(rng.uniform(lo, hi, size=size), 2)
    else:
        # Log-normal (default)
        return sample_lognormal(tx_type, size=size, config=amount_config, rng=rng)
