"""
bitmap_utils.py
===============
Shared utilities for BitmapBench Phase 1 simulations.

Provides:
  - Bitmap construction helpers (create, set_bit, to_int)
  - OpCounter for tracking logical bitmap operations
  - Cache-aware benchmarking (hot/cold with eviction)
  - Ground-truth query verification
  - Anti-bias controls (GC disable, randomized order)

Based on: "Bitmap Index Design and Evaluation" (Chan & Ioannidis, SIGMOD 1998)
"""

import gc
import math
import os
import random
import statistics
import sys
import time

# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# BITMAP CONSTRUCTION HELPERS
# ============================================================================

def create_empty_bitmaps(count, N):
    """Create 'count' zero-initialized bitmaps, each capable of holding N bits."""
    byte_count = (N + 7) // 8
    return [bytearray(byte_count) for _ in range(count)]


def set_bit(bitmap, pos):
    """Set bit at position 'pos' in the bytearray bitmap."""
    bitmap[pos >> 3] |= 1 << (pos & 7)


def to_int(bitmap):
    """Convert a bytearray bitmap to a Python arbitrary-precision integer."""
    return int.from_bytes(bitmap, byteorder='little')


def bitmaps_to_ints(bitmaps):
    """Convert a list of bytearray bitmaps to a list of Python integers."""
    return [to_int(b) for b in bitmaps]


def all_ones(N):
    """Return a bitmap integer with N bits set to 1."""
    return (1 << N) - 1


# ============================================================================
# OPERATION COUNTER
# ============================================================================

class OpCounter:
    """Counts logical bitmap operations (AND, OR, NOT, XOR) and scans separately."""
    __slots__ = ('and_ops', 'or_ops', 'not_ops', 'xor_ops', 'scans')

    def __init__(self):
        self.and_ops = 0
        self.or_ops = 0
        self.not_ops = 0
        self.xor_ops = 0
        self.scans = 0

    def total_ops(self):
        return self.and_ops + self.or_ops + self.not_ops + self.xor_ops

    def reset(self):
        self.and_ops = 0
        self.or_ops = 0
        self.not_ops = 0
        self.xor_ops = 0
        self.scans = 0

    def __repr__(self):
        return (f"OpCounter(AND={self.and_ops}, OR={self.or_ops}, "
                f"NOT={self.not_ops}, XOR={self.xor_ops}, scans={self.scans})")

    def copy(self):
        c = OpCounter()
        c.and_ops = self.and_ops
        c.or_ops = self.or_ops
        c.not_ops = self.not_ops
        c.xor_ops = self.xor_ops
        c.scans = self.scans
        return c


# ============================================================================
# VALUE DECOMPOSITION
# ============================================================================

def decompose_value(v, bases):
    """Decompose value v into multi-component digits.

    bases = [b1, b2, ..., bn] (low-order first).
    Returns digits = [d1, d2, ..., dn] where v = d1 + b1*d2 + b1*b2*d3 + ...
    """
    digits = []
    remaining = v
    for b in bases:
        digits.append(remaining % b)
        remaining //= b
    return digits


def recompose_value(digits, bases):
    """Recompose digits to value. Inverse of decompose_value."""
    v = 0
    multiplier = 1
    for d, b in zip(digits, bases):
        v += d * multiplier
        multiplier *= b
    return v


# ============================================================================
# N-COMPONENT RANGE INDEX BUILDER
# ============================================================================

def build_range_ncomp(data, bases, N=None):
    """Build an n-component range-encoded bitmap index.

    Args:
        data: list of integer values in [0, prod(bases)-1]
        bases: list [b1, b2, ..., bn] (low-order first)
        N: number of records (default: len(data))

    Returns:
        List of n components, each a list of (b_i - 1) Python integers.
        component[i][j] = bitmap where digit_i <= j, for j in [0, b_i-2].
        (B_{b_i-1} = all-ones, not stored.)
    """
    if N is None:
        N = len(data)
    n = len(bases)
    components = []

    for comp_idx in range(n):
        b = bases[comp_idx]
        bufs = create_empty_bitmaps(b - 1, N)
        components.append(bufs)

    for r, v in enumerate(data):
        digits = decompose_value(v, bases)
        for comp_idx in range(n):
            d = digits[comp_idx]
            b = bases[comp_idx]
            # Set bit in all range bitmaps B_j where j >= d (i.e., d <= j)
            for j in range(d, b - 1):
                set_bit(components[comp_idx][j], r)

    # Convert to integers
    return [bitmaps_to_ints(comp) for comp in components]


def build_equality_ncomp(data, bases, N=None):
    """Build an n-component equality-encoded bitmap index.

    Returns:
        List of n components, each a list of b_i Python integers.
        component[i][j] = bitmap where digit_i == j.
    """
    if N is None:
        N = len(data)
    n = len(bases)
    components = []

    for comp_idx in range(n):
        b = bases[comp_idx]
        bufs = create_empty_bitmaps(b, N)
        components.append(bufs)

    for r, v in enumerate(data):
        digits = decompose_value(v, bases)
        for comp_idx in range(n):
            d = digits[comp_idx]
            set_bit(components[comp_idx][d], r)

    return [bitmaps_to_ints(comp) for comp in components]


# ============================================================================
# RANGE BITMAP ACCESS HELPER
# ============================================================================

def get_range_leq_bitmap(component, digit, base, N):
    """Get the 'digit_i <= digit' bitmap from a range-encoded component.

    Returns (bitmap_int, scanned) where scanned=True if an actual bitmap was read.
    """
    if digit < 0:
        return 0, False
    if digit >= base - 1:
        return all_ones(N), False
    return component[digit], True


# ============================================================================
# CACHE-AWARE BENCHMARKING
# ============================================================================

# Cache eviction buffer: 32 MiB — touches every 64-byte cache line
EVICTION_BUFFER_SIZE = 32 * 1024 * 1024
_eviction_buffer = bytearray(EVICTION_BUFFER_SIZE)

# Default benchmark parameters
HOT_WARMUP = 30
HOT_ITERATIONS = 200
HOT_ROUNDS = 7
COLD_TRIALS = 25


def evict_cpu_cache():
    """Force CPU cache eviction by reading a 32 MiB buffer with stride 64."""
    _eviction_buffer[0] = (_eviction_buffer[0] + 1) & 0xFF
    return sum(_eviction_buffer[::64])


def benchmark_hot(fn, warmup=HOT_WARMUP, iterations=HOT_ITERATIONS, rounds=HOT_ROUNDS):
    """Measure query under hot-cache conditions. Returns median time in microseconds."""
    for _ in range(warmup):
        fn()

    gc.disable()
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iterations):
            fn()
        times.append((time.perf_counter() - t0) / iterations)
    gc.enable()

    return statistics.median(times) * 1_000_000


def benchmark_cold(fn, trials=COLD_TRIALS):
    """Measure query under cold-cache conditions. Returns median time in microseconds."""
    gc.disable()
    times = []
    for _ in range(trials):
        evict_cpu_cache()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    gc.enable()

    return statistics.median(times) * 1_000_000


# ============================================================================
# GROUND-TRUTH VERIFICATION
# ============================================================================

def ground_truth_query(data, N, qtype, v):
    """Evaluate query by scanning the original dataset. Used ONLY for verification."""
    result = 0
    for r in range(N):
        match = False
        val = data[r]
        if qtype == 'leq':
            match = val <= v
        elif qtype == 'lt':
            match = val < v
        elif qtype == 'gt':
            match = val > v
        elif qtype == 'geq':
            match = val >= v
        elif qtype == 'eq':
            match = val == v
        elif qtype == 'neq':
            match = val != v
        if match:
            result |= (1 << r)
    return result


# ============================================================================
# DECOMPOSITION ENUMERATION
# ============================================================================

def enumerate_decompositions(C, max_n=None):
    """Generate all valid base decompositions for cardinality C.

    Each decomposition is a tuple (b_1, b_2, ..., b_n) where:
      - b_i >= 2 for all i
      - b_1 <= b_2 <= ... <= b_n  (canonical ordering)
      - prod(b_i) >= C

    Returns list of tuples, sorted by (n_components, space).
    """
    if max_n is None:
        max_n = math.ceil(math.log2(C)) if C > 1 else 1

    results = []

    def _recurse(remaining_C, n_left, min_base, current):
        if n_left == 0:
            if remaining_C <= 1:
                results.append(tuple(current))
            return

        # The remaining components must each be >= min_base
        # and their product must be >= remaining_C
        max_base_needed = remaining_C  # single component covering remaining
        if n_left == 1:
            # Last component: must be >= remaining_C
            b = max(min_base, remaining_C)
            if b <= C:  # reasonable upper bound
                results.append(tuple(current + [b]))
            return

        # For n_left > 1 components, sweep base from min_base upward
        # Upper bound: if b^n_left >= remaining_C is already satisfied, no need to go higher
        # Also cap at remaining_C (no need for a component larger than remaining_C)
        upper = min(remaining_C, C)
        for b in range(min_base, upper + 1):
            new_remaining = math.ceil(remaining_C / b)
            if new_remaining < 2:
                new_remaining = 1
            # Check feasibility: can (n_left-1) components with min_base=b cover new_remaining?
            if b ** (n_left - 1) >= new_remaining:
                _recurse(new_remaining, n_left - 1, b, current + [b])
            elif new_remaining > b ** (n_left - 1):
                # Even with max base, can't cover. Skip to larger base.
                continue

    for n in range(1, max_n + 1):
        _recurse(C, n, 2, [])

    return results


def space_equality(bases):
    """Number of bitmaps for equality encoding: sum(b_i)."""
    return sum(bases)


def space_range(bases):
    """Number of bitmaps for range encoding: sum(b_i - 1)."""
    return sum(b - 1 for b in bases)


def time_equality(bases, C):
    """Expected bitmap scans per query for equality encoding (uniform, avg over ≤ and =).

    From Equation 2 of the paper:
    T(I) = (1/3C) * sum_i [ (C/B_i) * (2*b_i + 1) ]
    where B_i = prod(b_j for j < i)
    """
    n = len(bases)
    total = 0.0
    B_i = 1  # product of bases below component i
    for i in range(n):
        b_i = bases[i]
        total += (C / B_i) * (2 * b_i + 1) / 3.0
        B_i *= b_i
    return total / C


def time_range(bases):
    """Expected bitmap scans per query for range encoding (uniform, avg over ≤ and =).

    Adapted from Equation 4 of the paper.
    For the optimized RangeEval-Opt algorithm:
    T(I) = 2(n-1) + 1 - 2*sum(1/b_i) + (2/3)*sum(1 - 1/b_i)

    Simplified: average scans = sum over components of expected scans per component.
    For range encoding with RangeEval-Opt:
      - Component 1 (LSB): avg 1 - 1/b_1 scans (sometimes all-ones/zero = no scan)
      - Components 2..n: avg 2 - 2/b_i scans each
    Total: (1 - 1/b_1) + sum_{i=2}^{n} (2 - 2/b_i)
    """
    n = len(bases)
    if n == 1:
        return 1.0 - 1.0 / bases[0]

    total = 1.0 - 1.0 / bases[0]  # LSB component
    for i in range(1, n):
        total += 2.0 - 2.0 / bases[i]
    return total
