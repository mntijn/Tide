# Log-Normal Transaction Amount Distribution Changes

## Overview

All transaction amount generation was migrated from uniform distributions to **log-normal distributions** calibrated to real-world payment data. This affects both legitimate background patterns and fraud pattern camouflage amounts. Structuring behavior (amounts deliberately below reporting thresholds) is preserved unchanged.

## Motivation

Uniform distributions produce unrealistic transaction amounts: real-world payment amounts are heavily right-skewed with a log-normal body and Pareto (power-law) tail. Using uniform distributions creates a detectable artifact that models can exploit (data leakage) rather than learning genuine AML structural signals.

## Authoritative Sources

All calibration parameters are derived from the following sources:

1. **2022 Federal Reserve Payments Study** (8th Triennial, published March 2025)
   - Card payments: $9.76T across 153.3B transactions → mean ≈ $64
   - ACH transfers: $91.85T across 34.7B transactions → mean ≈ $2,642
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

## Calibrated Distribution Parameters

| Transaction Type | μ   | σ   | Median  | Min     | Max        | Source                          |
|-----------------|-----|-----|---------|---------|------------|---------------------------------|
| Payment         | 3.8 | 1.2 | ~$45    | $1      | $50,000    | Fed Payments Study (card)       |
| Transfer        | 5.5 | 2.0 | ~$245   | $5      | $500,000   | Fed Payments Study (ACH)        |
| Deposit         | 5.3 | 1.5 | ~$200   | $10     | $100,000   | FinCEN CTR data, Fed data       |
| Withdrawal      | 4.8 | 0.9 | ~$122   | $20     | $5,000     | Fed 2021 ATM avg $198, ATM limits |
| Salary          | 7.4 | 0.6 | ~$1,636 | $200    | $30,000    | Fed direct deposit averages     |
| High Value      | 9.6 | 1.5 | ~$14.9K | $5,000  | $1,000,000 | ACH large-value segment         |

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
