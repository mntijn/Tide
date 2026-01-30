"""
Temporal interleaving utilities for feature distribution matching.

This module provides functions to generate timestamps and temporal patterns
that overlap between fraud and legitimate patterns, preventing temporal
feature-based data leakage.

Key insight: Fraud patterns use "high_frequency" timestamps (all within 24 hours)
and generate rapid sequences with small `time_since_previous_transaction`.
Legitimate patterns need similar temporal density to prevent leakage.
"""
import datetime
from typing import List, Tuple, Optional
from .random_instance import random_instance


def generate_burst_timestamps(
    start_time: datetime.datetime,
    count: int,
    burst_duration_hours: float = 24.0,
    min_gap_minutes: int = 1,
    max_gap_minutes: int = 60,
) -> List[datetime.datetime]:
    """
    Generate timestamps clustered in a tight burst (mimics fraud "high_frequency").

    This creates temporal patterns similar to fraud patterns where many
    transactions occur within a short window.

    Args:
        start_time: Start of the burst window
        count: Number of timestamps to generate
        burst_duration_hours: Maximum duration of the burst (default 24 hours)
        min_gap_minutes: Minimum gap between transactions
        max_gap_minutes: Maximum gap between transactions

    Returns:
        Sorted list of timestamps within the burst window
    """
    timestamps = []
    max_minutes = int(burst_duration_hours * 60)

    for _ in range(count):
        offset_minutes = random_instance.randint(0, max_minutes)
        timestamps.append(
            start_time + datetime.timedelta(minutes=offset_minutes))

    return sorted(timestamps)


def generate_rapid_sequence_timestamps(
    start_time: datetime.datetime,
    count: int,
    gap_range_minutes: Tuple[int, int] = (5, 120),
) -> List[datetime.datetime]:
    """
    Generate timestamps with small, consistent gaps (mimics rapid fund movement).

    This creates sequences where `time_since_previous_transaction` is small,
    similar to fraud patterns.

    Args:
        start_time: Start time of the sequence
        count: Number of timestamps
        gap_range_minutes: (min, max) gap between consecutive transactions

    Returns:
        List of timestamps with small gaps
    """
    timestamps = [start_time]
    current_time = start_time

    for _ in range(count - 1):
        gap = random_instance.randint(
            gap_range_minutes[0], gap_range_minutes[1])
        current_time += datetime.timedelta(minutes=gap)
        timestamps.append(current_time)

    return timestamps


def generate_interleaved_timestamps(
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    count: int,
    burst_probability: float = 0.3,
    burst_size_range: Tuple[int, int] = (3, 10),
) -> List[datetime.datetime]:
    """
    Generate timestamps that mix spread-out and burst patterns.

    This prevents temporal patterns from being too distinctive by mixing:
    - Regular spread-out timestamps (legitimate-like)
    - Burst timestamps (fraud-like)

    Args:
        start_date: Start of time span
        end_date: End of time span
        count: Total number of timestamps to generate
        burst_probability: Probability of generating a burst vs spread
        burst_size_range: (min, max) size of bursts when they occur

    Returns:
        Sorted list of timestamps
    """
    timestamps = []
    remaining = count
    total_days = max(1, (end_date - start_date).days)

    while remaining > 0:
        if random_instance.random() < burst_probability and remaining >= burst_size_range[0]:
            # Generate a burst
            burst_size = min(
                remaining,
                random_instance.randint(
                    burst_size_range[0], burst_size_range[1])
            )
            burst_day = random_instance.randint(0, total_days - 1)
            burst_start = start_date + datetime.timedelta(
                days=burst_day,
                hours=random_instance.randint(8, 18)
            )
            burst_timestamps = generate_burst_timestamps(
                burst_start, burst_size, burst_duration_hours=12
            )
            timestamps.extend(burst_timestamps)
            remaining -= burst_size
        else:
            # Generate a single spread-out timestamp
            random_day = random_instance.randint(0, total_days - 1)
            random_hour = random_instance.randint(0, 23)
            random_minute = random_instance.randint(0, 59)
            ts = start_date + datetime.timedelta(
                days=random_day, hours=random_hour, minutes=random_minute
            )
            timestamps.append(ts)
            remaining -= 1

    return sorted(timestamps)


def generate_deposit_transfer_sequence(
    start_time: datetime.datetime,
    num_cycles: int,
    deposit_transfer_delay_hours: Tuple[float, float] = (0.5, 6.0),
    cycle_gap_hours: Tuple[float, float] = (12.0, 72.0),
) -> List[Tuple[datetime.datetime, str]]:
    """
    Generate timestamp pairs for deposit→transfer sequences.

    This mimics the fraud pattern of deposit followed by immediate transfer,
    but for legitimate purposes (business operations, bill payments).

    Args:
        start_time: Start of the sequence
        num_cycles: Number of deposit→transfer pairs
        deposit_transfer_delay_hours: Delay between deposit and transfer
        cycle_gap_hours: Gap between cycles

    Returns:
        List of (timestamp, type) tuples where type is "deposit" or "transfer"
    """
    sequence = []
    current_time = start_time

    for _ in range(num_cycles):
        # Deposit
        sequence.append((current_time, "deposit"))

        # Transfer after delay
        delay = random_instance.uniform(
            deposit_transfer_delay_hours[0],
            deposit_transfer_delay_hours[1]
        )
        transfer_time = current_time + datetime.timedelta(hours=delay)
        sequence.append((transfer_time, "transfer"))

        # Gap to next cycle
        gap = random_instance.uniform(cycle_gap_hours[0], cycle_gap_hours[1])
        current_time = transfer_time + datetime.timedelta(hours=gap)

    return sequence


def estimate_time_since_previous_distribution(
    timestamps: List[datetime.datetime],
) -> dict:
    """
    Estimate the time_since_previous_transaction distribution for a set of timestamps.

    Useful for debugging and verifying feature overlap.

    Args:
        timestamps: Sorted list of timestamps

    Returns:
        Dictionary with distribution statistics
    """
    if len(timestamps) < 2:
        return {"min": None, "max": None, "mean": None, "count": len(timestamps)}

    gaps = []
    sorted_ts = sorted(timestamps)
    for i in range(1, len(sorted_ts)):
        gap = (sorted_ts[i] - sorted_ts[i - 1]).total_seconds() / 60  # minutes
        gaps.append(gap)

    return {
        "min_minutes": min(gaps),
        "max_minutes": max(gaps),
        "mean_minutes": sum(gaps) / len(gaps),
        "median_minutes": sorted(gaps)[len(gaps) // 2],
        "count": len(timestamps),
        "under_60min": sum(1 for g in gaps if g < 60),
        "under_24h": sum(1 for g in gaps if g < 1440),
    }
