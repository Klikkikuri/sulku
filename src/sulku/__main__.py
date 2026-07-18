"""
Main Execution Entry Point
==========================

This module enables execution of the package as a script using `python -m sulku`.
It imports and runs the Click CLI command group.
"""

from sulku.cli import main

if __name__ == "__main__":
    main()
