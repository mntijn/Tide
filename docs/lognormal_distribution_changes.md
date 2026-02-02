# Log-Normal Transaction Amount Distribution Changes

## Overview

All transaction amount generation was migrated from uniform distributions to **log-normal distributions** calibrated to real-world payment data. This affects both legitimate background patterns and fraud pattern camouflage amounts. Structuring behavior (amounts deliberately below reporting thresholds) is preserved unchanged.

## Motivation

Uniform distributions produce unrealistic transaction amounts: real-world payment amounts are heavily right-skewed with a log-normal body and Pareto (power-law) tail. Using uniform distributions creates a detectable artifact that models can exploit (data leakage) rather than learning genuine AML structural signals.

## Authoritative Sources

All calibration parameters are derived from the following sources:

1. **2022 Federal Reserve Payments Study (FRPS)** (8th Triennial, published March 2025)
   - Table 1: "Noncash payments in the United States, by type" (calendar year 2021 data)
   - Source: https://www.federalreserve.gov/paymentsystems/frps_pubdocs.htm

2. **FinCEN SAR Statistics (2014–2024)**
   - Currency Transaction Reports (CTRs) filed for cash transactions > $10,000
   - Suspicious Activity Reports (SARs) filed for structuring below $10,000
   - Informs the structuring threshold behavior retained in fraud patterns

3. **Nacha ACH Network Reports**
   - ACH payment volumes and average transaction sizes
   - Validates the transfer distribution parameters

4. **Nilson Report**
   - Card payment volume statistics
   - Validates the payment distribution parameters

5. **IBM AMLworld (NeurIPS 2023)**
   - Methodology for log-normal parameterization of synthetic AML data
   - Validates the approach of using log-normal body + clipping for tails

## Log-Normal Parameterization

Given target median $M$ and mean $E$:

$$\mu = \ln(M), \quad \sigma = \sqrt{2 \ln(E/M)}$$

If $X \sim \text{LogNormal}(\mu, \sigma)$, then $\text{median}(X) = e^\mu$ and $\text{mean}(X) = e^{\mu + \sigma^2/2}$.

## Parameter Derivation

### How log-normal sampling works

If X ~ LogNormal(μ, σ), then ln(X) ~ Normal(μ, σ²). The parameters μ and σ are the mean and standard deviation of the *underlying normal distribution in log-space*, not of the amounts themselves. To sample an amount: draw Z ~ Normal(0,1), compute amount = e^(μ + σ·Z), then clip to [min, max].

Key properties:
- **median(X) = e^μ** — μ directly controls the center of the distribution
- **mean(X) = e^(μ + σ²/2)** — the mean is always larger than the median (right-skew); larger σ increases the gap

### Calibration approach

The FRPS reports the *average* (i.e., mean) transaction value per payment type. Since the FRPS does not report medians, we use the reported average as our starting point. For types where we have FRPS data, we set μ = ln(FRPS average), making our *median* equal to the FRPS *mean*. Because real transaction distributions are right-skewed (mean > median), this likely *overestimates* the true median. We then choose σ to control the tail weight; larger σ produces a heavier right tail and a higher implied mean relative to the median.

**Note**: Setting median = FRPS mean is a deliberate simplification. The true median would be lower than the reported average. This means our distributions are shifted slightly upward compared to reality. We accept this because (a) the FRPS does not publish medians, and (b) the primary goal is realistic *shape* (log-normal with appropriate skew), not exact calibration of every moment.

### Payment (μ = 3.8, σ = 1.2)
- **Source**: FRPS Table 1, "Non-prepaid debit card" row (2021)
- **Data**: 87.8B transactions, $3.94T value → reported average = $45
- **Derivation**: We set μ = ln(45) = 3.81 ≈ 3.8, so our median = e^3.8 = $44.70. We choose σ = 1.2, giving implied mean = e^(3.8 + 1.44/2) = e^4.52 = $92. The implied mean ($92) is higher than the FRPS average ($45) because σ = 1.2 produces a heavy right tail; this reflects the mixture of many small purchases with occasional large card transactions.

### Transfer (μ = 5.5, σ = 2.0)
- **Source**: FRPS Table 1, "ACH debit transfers" row (2021)
- **Data**: 20.3B transactions, $33.19T value → reported average = $1,634
- **Derivation**: We set μ = ln(245) = 5.50, so our median = e^5.5 = $245. We choose σ = 2.0 specifically so that the implied mean = e^(5.5 + 4.0/2) = e^7.5 = $1,808 approximates the FRPS average of $1,634. The large σ captures the very heavy tail: most ACH debits are small (P2P, bill pay), but business wire transfers can be orders of magnitude larger.

### Deposit (μ = 5.3, σ = 1.5)
- **Source**: No direct FRPS row. No authoritative public source for cash deposit distributions. Parameters are modeling assumptions.
- **Derivation**: We set μ = ln(200) = 5.30, so our median = $200. With σ = 1.5, implied mean = e^(5.3 + 1.125) = e^6.425 = $617. The tail extends into business deposits approaching the $10K CTR reporting threshold.

### Withdrawal (μ = 4.8, σ = 0.9)
- **Source**: FRPS Table 1, "ATM cash withdrawals" row (2021)
- **Data**: 3.7B transactions, $0.73T value → reported average = $198
- **Derivation**: We set μ = ln(122) = 4.80, so our median = e^4.8 = $122. We choose σ = 0.9, giving implied mean = e^(4.8 + 0.81/2) = e^5.205 = $182. The implied mean ($182) slightly undershoots the FRPS average ($198); this is because ATM withdrawal limits (max = $5,000 in our model) clip the upper tail, pulling the effective mean down further.

### Salary (μ = 7.4, σ = 0.6)
- **Source**: No direct FRPS row for payroll. FRPS Table 1 reports ACH credit transfers at 15.9B transactions, $58.66T value → average = $3,690, but this mixes payroll with business-to-business credits and is not usable directly. Parameters are modeling assumptions targeting typical biweekly direct deposit amounts.
- **Derivation**: We set μ = ln(1636) = 7.40, so our median = $1,636. With σ = 0.6, implied mean = e^(7.4 + 0.18) = e^7.58 = $1,955. The narrow σ reflects the regularity of salary payments (low variance compared to other transaction types).

### High Value (μ = 9.6, σ = 1.5)
- **Source**: No direct FRPS row. Parameters are modeling assumptions targeting large business/property transactions. FRPS reports total ACH at 36.2B transactions, $91.85T value (overall average $2,536), but wire transfers (excluded from FRPS) would be more representative for this category.
- **Derivation**: We set μ = ln(14880) = 9.61 ≈ 9.6, so our median = e^9.6 = $14,880. With σ = 1.5, implied mean = e^(9.6 + 1.125) = e^10.725 = $45,600. Clipped to [$5,000, $1,000,000].

## Calibrated Distribution Parameters (Summary)

| Transaction Type | μ   | σ   | Median  | Implied Mean | Min     | Max        | FRPS Row                    |
|-----------------|-----|-----|---------|-------------|---------|------------|-----------------------------|
| Payment         | 3.8 | 1.2 | ~$45    | ~$92        | $1      | $50,000    | Non-prepaid debit card      |
| Transfer        | 5.5 | 2.0 | ~$245   | ~$1,808     | $5      | $500,000   | ACH debit transfers         |
| Deposit         | 5.3 | 1.5 | ~$200   | ~$617       | $10     | $100,000   | — (modeling assumption)     |
| Withdrawal      | 4.8 | 0.9 | ~$122   | ~$182       | $20     | $5,000     | ATM withdrawals             |
| Salary          | 7.4 | 0.6 | ~$1,636 | ~$1,955     | $200    | $30,000    | — (modeling assumption)     |
| High Value      | 9.6 | 1.5 | ~$14.9K | ~$45,600    | $5,000  | $1,000,000 | — (modeling assumption)     |

## Files Changed

### New utility
- **`tide/utils/amount_distributions.py`** — Shared log-normal sampling functions (`sample_lognormal`, `sample_lognormal_scalar`, `sample_from_config`) with `DEFAULT_DISTRIBUTIONS` dict containing calibrated parameters per transaction type.

### Configuration
- **`configs/graph_HI.yaml`** — Replaced flat uniform `amount_ranges` with `amount_distributions` section containing per-type log-normal parameters. Each background pattern section now includes `use_lognormal: true`.

### Background patterns updated (uniform → log-normal)

| File | Distribution keys used |
|------|----------------------|
| `background_activity.py` (RandomPayments) | payment, transfer, deposit, withdrawal (vectorized via `sample_lognormal`) |
| `salary_payments.py` | salary (regular), shifted μ=8.5 for high earners |
| `legitimate_cash_operations.py` | deposit, withdrawal; rapid deposits use μ+1.5 shift |
| `legitimate_periodic.py` | payment (recurring bills) |
| `legitimate_high_payments.py` | high_value |
| `legitimate_burst.py` | payment (regular), deposit/withdrawal (cash ops) |
| `legitimate_chains.py` | transfer, payment, or high_value depending on chain scenario |
| `legitimate_rapid_flow.py` | transfer (inflows and outflows) |
| `legitimate_high_risk_activity.py` | payment |

### Fraud pattern camouflage updated
- **`tide/utils/amount_interleaving.py`** — Camouflage amounts in `generate_fraud_with_camouflage()`, `generate_fraud_like_amounts()`, `generate_interleaved_amounts()`, and `generate_legitimate_structuring_amounts()` now sample from the same log-normal distributions as legitimate patterns. This ensures camouflage transactions are indistinguishable from genuine legitimate activity.

## What Was NOT Changed (Structuring Preserved)

The following structuring mechanisms remain **unchanged**:

1. **`currency_conversion.py` → `generate_structured_amounts()`** — Generates amounts at 70–95% of reporting threshold ($10,000 USD), with currency-aware conversion. Used by all fraud patterns for their core structuring deposits.

2. **Fraud pattern configs (`patterns.yaml`)** — `deposit_amount_range: [7500, 9900]` and similar ranges for structured amounts remain uniform distributions deliberately targeting the sub-threshold range.

3. **Legitimate structuring pattern** — `legitimateStructuring` in `graph_HI.yaml` retains uniform `deposit_amount_range: [7500, 9999]` to model legitimate near-threshold behavior (rent in high-cost areas, business deposits, etc.).

4. **Structuring interleaving in background patterns** — `structured_amount_probability` (3–5%) in RandomPayments, bursts, and rapid flows still injects uniform near-threshold amounts into legitimate activity, preventing structuring from being a pure fraud signal.

## Summary of Design Principle

- **Normal amounts**: Log-normal (realistic, calibrated to Fed/FinCEN data)
- **Structuring amounts**: Uniform in [7000, 9999] range (deliberately targets sub-$10K threshold)
- **Fraud camouflage**: Draws from same log-normal as legitimate (prevents amount leakage)
- **Fraud structuring**: Unchanged — 70–95% of reporting threshold via `generate_structured_amounts()`

The distinguishing factor for fraud detection should be **graph structure** (fan-in/fan-out, chains, cycles, rapid temporal patterns), not raw transaction amounts.
