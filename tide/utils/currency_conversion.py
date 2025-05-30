"""
Currency conversion utility for AML pattern structuring.

This module provides efficient currency conversion using predefined exchange rates
to ensure transaction amounts stay below reporting thresholds when converted
to the reporting currency.

Exchange rates are based on data from May 30, 2025 from Federal Reserve H.10
and other authoritative sources.
"""

import datetime
from typing import Dict, Tuple, Optional
from .constants import COUNTRY_TO_CURRENCY


class CurrencyConverter:
    """Currency converter with predefined exchange rates for efficiency."""

    # Exchange rates as of May 30, 2025 (1 USD = X foreign currency)
    # Source: Federal Reserve H.10, OANDA, and other financial data providers
    USD_EXCHANGE_RATES = {
        'USD': 1.0000,      # Base currency
        'EUR': 0.8814,      # Euro (European Union)
        'GBP': 0.7404,      # British Pound Sterling
        'JPY': 142.61,      # Japanese Yen
        'CHF': 0.8214,      # Swiss Franc
        'AED': 3.6725,      # UAE Dirham
        'HKD': 7.8317,      # Hong Kong Dollar
        'SGD': 1.2843,      # Singapore Dollar
        'BSD': 1.0000,      # Bahamian Dollar (pegged to USD)
        'SCR': 13.7500,     # Seychellois Rupee
        'BBD': 2.0000,      # Barbadian Dollar (pegged to USD)
        'BMD': 1.0000,      # Bermudian Dollar (pegged to USD)
        'BZD': 2.0150,      # Belize Dollar
        'VUV': 118.50,      # Vanuatu Vatu
        'XCD': 2.7000,      # East Caribbean Dollar
    }

    def __init__(self):
        """Initialize the currency converter."""
        # Create reverse mapping for efficiency (foreign currency to USD)
        self.to_usd_rates = {}
        for currency, rate in self.USD_EXCHANGE_RATES.items():
            if rate != 0:
                self.to_usd_rates[currency] = 1.0 / rate
            else:
                self.to_usd_rates[currency] = 0.0

    def convert(self, amount: float, from_currency: str, to_currency: str) -> float:
        """
        Convert amount from one currency to another.

        Args:
            amount: Amount to convert
            from_currency: Source currency code
            to_currency: Target currency code

        Returns:
            Converted amount in target currency

        Raises:
            ValueError: If currency codes are not supported
        """
        if from_currency == to_currency:
            return amount

        # Validate currency codes
        if from_currency not in self.USD_EXCHANGE_RATES:
            raise ValueError(f"Unsupported source currency: {from_currency}")
        if to_currency not in self.USD_EXCHANGE_RATES:
            raise ValueError(f"Unsupported target currency: {to_currency}")

        # Convert via USD as base
        if from_currency == 'USD':
            # USD to target currency
            return round(amount * self.USD_EXCHANGE_RATES[to_currency], 2)
        elif to_currency == 'USD':
            # Source currency to USD
            return round(amount * self.to_usd_rates[from_currency], 2)
        else:
            # Source to USD, then USD to target
            amount_in_usd = amount * self.to_usd_rates[from_currency]
            return round(amount_in_usd * self.USD_EXCHANGE_RATES[to_currency], 2)

    def get_supported_currencies(self) -> list:
        """Get list of supported currency codes."""
        return list(self.USD_EXCHANGE_RATES.keys())

    def get_rate(self, from_currency: str, to_currency: str) -> float:
        """
        Get exchange rate between two currencies.

        Args:
            from_currency: Source currency code
            to_currency: Target currency code

        Returns:
            Exchange rate (1 unit of from_currency = X units of to_currency)
        """
        return self.convert(1.0, from_currency, to_currency)

    def is_supported(self, currency_code: str) -> bool:
        """Check if a currency code is supported."""
        return currency_code in self.USD_EXCHANGE_RATES


class StructuringAmountGenerator:
    """Generate transaction amounts structured to avoid reporting thresholds."""

    def __init__(self, converter: CurrencyConverter):
        """
        Initialize with a currency converter.

        Args:
            converter: CurrencyConverter instance
        """
        self.converter = converter

    def generate_structured_amounts(
        self,
        count: int,
        reporting_threshold: float,
        reporting_currency: str,
        target_currency: str = None,
        base_amount: float = None,
        variation_pct: float = 0.15
    ) -> list:
        """
        Generate amounts structured to avoid reporting thresholds.

        Args:
            count: Number of amounts to generate
            reporting_threshold: Threshold amount in reporting currency (set in graph.yaml)
            reporting_currency: Currency code for reporting (set in graph.yaml)
            target_currency: Currency for the transaction (defaults to reporting_currency)
            base_amount: Base amount to use (defaults to 70-95% of threshold)
            variation_pct: Percentage variation to add (default 15%)

        Returns:
            List of structured amounts in target currency
        """
        from .random_instance import random_instance

        target_currency = target_currency or reporting_currency

        # Convert threshold to target currency if different
        if target_currency != reporting_currency:
            threshold_in_target = self.converter.convert(
                reporting_threshold, reporting_currency, target_currency
            )
        else:
            threshold_in_target = reporting_threshold

        # Set base amount if not provided
        if base_amount is None:
            base_amount = threshold_in_target * \
                round(random_instance.uniform(0.70, 0.95), 2)

        amounts = []
        for _ in range(count):
            # Add variation to avoid exact patterns
            variation = random_instance.uniform(-base_amount *
                                                variation_pct, base_amount * variation_pct)
            structured_amount = max(100.0, base_amount + variation)

            # Ensure amount stays below threshold when converted to reporting currency
            if target_currency != reporting_currency:
                amount_in_reporting = self.converter.convert(
                    structured_amount, target_currency, reporting_currency
                )
                # Scale down if it exceeds threshold
                if amount_in_reporting >= reporting_threshold:
                    scale_factor = (reporting_threshold *
                                    0.95) / amount_in_reporting
                    structured_amount = structured_amount * scale_factor

            amounts.append(round(structured_amount, 2))

        return amounts


# Global instances for convenience
_converter = CurrencyConverter()
_structuring_generator = StructuringAmountGenerator(_converter)


def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    """
    Convenience function to convert currency using the global converter.

    Args:
        amount: Amount to convert
        from_currency: Source currency code
        to_currency: Target currency code

    Returns:
        Converted amount
    """
    return _converter.convert(amount, from_currency, to_currency)


def generate_structured_amounts(
    count: int,
    reporting_threshold: float,
    reporting_currency: str,
    target_currency: str = None,
    base_amount: float = None
) -> list:
    """
    Convenience function to generate structured amounts.

    Args:
        count: Number of amounts to generate
        reporting_threshold: Threshold in reporting currency
        reporting_currency: Reporting currency code
        target_currency: Transaction currency code
        base_amount: Base amount (optional)

    Returns:
        List of structured amounts
    """
    return _structuring_generator.generate_structured_amounts(
        count=count,
        reporting_threshold=reporting_threshold,
        reporting_currency=reporting_currency,
        target_currency=target_currency,
        base_amount=base_amount
    )


def get_supported_currencies() -> list:
    """Get list of supported currency codes."""
    return _converter.get_supported_currencies()


def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """Get exchange rate between two currencies."""
    return _converter.get_rate(from_currency, to_currency)


# Validation function for currency codes used in the system
def validate_system_currencies():
    """
    Validate that all currencies used in COUNTRY_TO_CURRENCY are supported.

    Returns:
        Tuple of (supported, unsupported) currency lists
    """
    system_currencies = set(COUNTRY_TO_CURRENCY.values())
    supported = []
    unsupported = []

    for currency in system_currencies:
        if _converter.is_supported(currency):
            supported.append(currency)
        else:
            unsupported.append(currency)

    return supported, unsupported


if __name__ == "__main__":
    # Example usage and validation
    print("=== Currency Conversion Utility ===")
    print(f"Supported currencies: {get_supported_currencies()}")
    print()

    # Validate system currencies
    supported, unsupported = validate_system_currencies()
    print(f"System currencies supported: {supported}")
    if unsupported:
        print(f"System currencies NOT supported: {unsupported}")
    print()

    # Example conversions
    print("Example conversions (as of May 30, 2025):")
    examples = [
        (10000, 'USD', 'EUR'),
        (10000, 'USD', 'GBP'),
        (10000, 'USD', 'JPY'),
        (8814, 'EUR', 'USD'),
        (7404, 'GBP', 'USD'),
    ]

    for amount, from_curr, to_curr in examples:
        converted = convert_currency(amount, from_curr, to_curr)
        print(f"{amount:,.2f} {from_curr} = {converted:,.2f} {to_curr}")

    print()

    # Example structured amounts
    print("Example structured amounts (below $10,000 USD):")
    structured = generate_structured_amounts(
        count=5,
        reporting_threshold=10000,
        reporting_currency='USD',
        target_currency='EUR'
    )

    for i, amount in enumerate(structured, 1):
        amount_usd = convert_currency(amount, 'EUR', 'USD')
        status = "✓" if amount_usd < 10000 else "✗"
        print(f"  {i}. {amount:>8.2f} EUR → {amount_usd:>8.2f} USD {status}")
