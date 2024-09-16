# run_all_tests.py
import unittest
import argparse
from tabulate import tabulate
from termcolor import colored

class CustomTestResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.successes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)

class CustomTestRunner(unittest.TextTestRunner):
    resultclass = CustomTestResult
    
    def run(self, test):
        result = super().run(test)
        self.print_report(result)
        return result
    
    def print_report(self, result):
        results_by_file = {}
        
        # Aggregate results by file and class
        for test, err in result.failures + result.errors:
            module = test.__class__.__module__.replace('.', '/') + '.py'  # Get the file name with .py extension
            class_name = test.__class__.__name__
            friendly_class_name = test.__class__.__doc__ or class_name
            method_name = test._testMethodName
            friendly_method_name = test._testMethodDoc or method_name

            if module not in results_by_file:
                results_by_file[module] = {}
            if friendly_class_name not in results_by_file[module]:
                results_by_file[module][friendly_class_name] = {'total': 0, 'passed': [], 'failed': []}
            results_by_file[module][friendly_class_name]['total'] += 1
            results_by_file[module][friendly_class_name]['failed'].append(colored(friendly_method_name, 'red'))
        
        for test in result.successes:
            module = test.__class__.__module__.replace('.', '/') + '.py'  # Get the file name with .py extension
            class_name = test.__class__.__name__
            friendly_class_name = test.__class__.__doc__ or class_name
            method_name = test._testMethodName
            friendly_method_name = test._testMethodDoc or method_name

            if module not in results_by_file:
                results_by_file[module] = {}
            if friendly_class_name not in results_by_file[module]:
                results_by_file[module][friendly_class_name] = {'total': 0, 'passed': [], 'failed': []}
            results_by_file[module][friendly_class_name]['total'] += 1
            results_by_file[module][friendly_class_name]['passed'].append(colored(friendly_method_name, 'green'))

        # Print results for each file
        for module, classes in results_by_file.items():
            table_data = []
            for class_name, stats in classes.items():
                tests = stats['failed'] + stats['passed']
                table_data.append([class_name, "\n".join(tests)])
            
            print(f"\n\n{module}")
            print(tabulate(table_data, headers=["Class", "Tests"], tablefmt="grid"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run test cases for RL programming assignments.")
    parser.add_argument('assignment', choices=['bandit'], help="Select an assignment to run tests for")
    args = parser.parse_args()

    loader = unittest.TestLoader()
    suite = loader.discover('./tests', pattern=f'{args.assignment}.py')
    
    runner = CustomTestRunner()
    runner.run(suite)
