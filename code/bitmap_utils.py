"""
bitmap_utils.py
===============
Shared bitmap utilities for BitmapBench simulations.
Canonical implementation is located in `code/common/bitmap_utils.py`.
"""

import os
import sys

_common_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'common')
if _common_dir not in sys.path:
    sys.path.insert(0, _common_dir)

from common.bitmap_utils import *
