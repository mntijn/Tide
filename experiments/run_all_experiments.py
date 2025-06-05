#!/usr/bin/env python3
"""
Master Experimental Validation Script

Runs all four hypothesis tests to validate Tide's functionality:
- H1: Pattern generation validation
- H2: Scalability testing
- H3: Configuration compliance
- H4: Reproducibility testing
"""

import sys
import time
from pathlib import Path

# Import all experiment modules
sys.path.append(str(Path(__file__).parent))

try:
    from h1_pattern_validation import run_h1_experiment
    from h2_scalability import run_h2_experiment
    from h3_config_compliance import run_h3_experiment
    from h4_reproducibility import run_h4_experiment
except ImportError as e:
    print(f"Error importing experiment modules: {e}")
    sys.exit(1)


def run_all_experiments():
    """Run all experimental validations"""
    print("=" * 60)
    print("TIDE EXPERIMENTAL VALIDATION SUITE")
    print("=" * 60)
    print()

    experiments = [
        ("H1: Pattern Validation", run_h1_experiment),
        ("H2: Scalability Testing", run_h2_experiment),
        ("H3: Configuration Compliance", run_h3_experiment),
        ("H4: Reproducibility Testing", run_h4_experiment)
    ]

    results = {}
    total_start_time = time.time()

    for name, experiment_func in experiments:
        print(f"Starting {name}...")
        start_time = time.time()

        try:
            success = experiment_func()
            duration = time.time() - start_time
            results[name] = {
                'success': success,
                'duration': duration,
                'error': None
            }

            status = "PASSED" if success else "FAILED"
            print(f"{name} {status} (took {duration:.1f}s)")

        except Exception as e:
            duration = time.time() - start_time
            results[name] = {
                'success': False,
                'duration': duration,
                'error': str(e)
            }

            print(f"{name} ERROR: {e} (took {duration:.1f}s)")

        print("-" * 60)
        print()

    total_duration = time.time() - total_start_time

    # Print summary
    print("=" * 60)
    print("EXPERIMENTAL VALIDATION SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for r in results.values() if r['success'])
    total_count = len(results)

    for name, result in results.items():
        status = "PASSED" if result['success'] else "FAILED"
        duration = result['duration']
        print(f"{name:<30} {status:<8} ({duration:.1f}s)")

        if result['error']:
            print(f"  Error: {result['error']}")

    print("-" * 60)
    print(f"Total: {passed_count}/{total_count} experiments passed")
    print(f"Total runtime: {total_duration:.1f}s")

    if passed_count == total_count:
        print("\n✓ All experimental validations PASSED")
        print("✓ Tide meets all design requirements")
        return True
    else:
        print(
            f"\n✗ {total_count - passed_count} experimental validation(s) FAILED")
        print("✗ Tide has issues that need to be addressed")
        return False


def main():
    """Main entry point"""
    success = run_all_experiments()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
