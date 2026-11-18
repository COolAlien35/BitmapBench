"""
bitmap_utils.py
===============
Shared bitmap utilities for BitmapBench simulations.
Canonical implementation is located in `code/common/bitmap_utils.py`.
"""

import os
import sys

COMMON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'common')
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from bitmap_utils import *
