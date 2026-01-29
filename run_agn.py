#!/usr/bin/env python3
"""
Main entry point for Generalized AGN
Run this script from the project root directory
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agn_general.main import main

if __name__ == "__main__":
    main()
