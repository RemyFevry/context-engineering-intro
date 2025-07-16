#!/usr/bin/env python3
"""
Test runner for job application tracking system.

Provides convenient ways to run different types of tests with proper
environment setup and reporting.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description or cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    return result.returncode == 0


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run tests for job application tracking system")
    
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--slow', action='store_true', help='Include slow tests')
    parser.add_argument('--api', action='store_true', help='Include API tests (requires API keys)')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage reporting')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fast', action='store_true', help='Run fast tests only')
    parser.add_argument('--file', '-f', help='Run specific test file')
    parser.add_argument('--function', '-k', help='Run specific test function/pattern')
    parser.add_argument('--parallel', '-n', type=int, help='Run tests in parallel (requires pytest-xdist)')
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd_parts = ['python', '-m', 'pytest']
    
    # Add verbosity
    if args.verbose:
        cmd_parts.append('-v')
    
    # Add coverage
    if args.coverage:
        cmd_parts.extend(['--cov=.', '--cov-report=html', '--cov-report=term-missing'])
    
    # Add parallel execution
    if args.parallel:
        cmd_parts.extend(['-n', str(args.parallel)])
    
    # Add test selection
    if args.unit:
        cmd_parts.extend(['-m', 'unit'])
    elif args.integration:
        cmd_parts.extend(['-m', 'integration'])
    elif args.fast:
        cmd_parts.extend(['-m', 'not slow and not api'])
    elif args.api:
        cmd_parts.extend(['-m', 'api'])
    elif not args.slow:
        cmd_parts.extend(['-m', 'not slow'])
    
    # Add specific file
    if args.file:
        cmd_parts.append(f'tests/{args.file}')
    
    # Add specific function
    if args.function:
        cmd_parts.extend(['-k', args.function])
    
    # Run pre-test checks
    print("Job Application Tracking System - Test Runner")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('tests').exists():
        print("Error: tests directory not found. Run from project root.")
        sys.exit(1)
    
    # Check if requirements are installed
    try:
        import pytest
        print("✓ pytest is installed")
    except ImportError:
        print("✗ pytest is not installed. Run: pip install pytest")
        sys.exit(1)
    
    # Check for optional dependencies
    try:
        import pytest_asyncio
        print("✓ pytest-asyncio is available")
    except ImportError:
        print("⚠ pytest-asyncio not found. Install with: pip install pytest-asyncio")
    
    try:
        import pytest_cov
        print("✓ pytest-cov is available")
    except ImportError:
        if args.coverage:
            print("✗ pytest-cov not found but coverage requested. Install with: pip install pytest-cov")
            sys.exit(1)
        else:
            print("⚠ pytest-cov not found. Install with: pip install pytest-cov")
    
    # Run tests
    cmd = ' '.join(cmd_parts)
    success = run_command(cmd, f"Running tests with: {cmd}")
    
    # Summary
    print(f"\n{'='*60}")
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed.")
        
    print(f"{'='*60}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()