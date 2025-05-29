import unittest
import sys
import json
from datetime import datetime
from test_graph_generator import TestGraphGenerator


def run_tests_and_generate_report():
    """Run tests and generate a detailed report"""
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGraphGenerator)

    # Create a test runner that will collect results
    class TestResultCollector(unittest.TestResult):
        def __init__(self):
            super().__init__()
            self.test_results = []
            self.start_time = None
            self.end_time = None

        def startTest(self, test):
            self.start_time = datetime.now()
            super().startTest(test)

        def stopTest(self, test):
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()

            result = {
                'test_name': test._testMethodName,
                'docstring': test._testMethodDoc,
                'duration': duration,
                'status': 'PASS' if not self.failures and not self.errors else 'FAIL',
                'error': None
            }

            # Add error message if test failed
            for failure in self.failures:
                if failure[0] == test:
                    result['error'] = failure[1]
            for error in self.errors:
                if error[0] == test:
                    result['error'] = error[1]

            self.test_results.append(result)
            super().stopTest(test)

    # Run the tests
    collector = TestResultCollector()
    suite.run(collector)

    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': len(collector.test_results),
        'passed_tests': len([r for r in collector.test_results if r['status'] == 'PASS']),
        'failed_tests': len([r for r in collector.test_results if r['status'] == 'FAIL']),
        'total_duration': sum(r['duration'] for r in collector.test_results),
        'test_results': collector.test_results
    }

    # Print summary
    print("\n=== Test Summary ===")
    print(f"Total tests: {report['total_tests']}")
    print(f"Passed: {report['passed_tests']}")
    print(f"Failed: {report['failed_tests']}")
    print(f"Total duration: {report['total_duration']:.2f} seconds")

    # Print detailed results
    print("\n=== Detailed Results ===")
    for result in collector.test_results:
        status_symbol = "✓" if result['status'] == 'PASS' else "✗"
        print(
            f"{status_symbol} {result['test_name']} ({result['duration']:.2f}s)")
        if result['error']:
            print(f"   Error: {result['error']}")

    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to: {report_file}")

    # Return appropriate exit code
    return 0 if report['failed_tests'] == 0 else 1


if __name__ == '__main__':
    sys.exit(run_tests_and_generate_report())
