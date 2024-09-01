import unittest
import os
import sys

# Add the parent directory to the Python path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_tests():
    # Discover and run tests in the 'tests' directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Return the number of failures and errors
    return len(result.failures) + len(result.errors)

if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)