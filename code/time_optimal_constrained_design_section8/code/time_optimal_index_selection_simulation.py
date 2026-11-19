"""
time_optimal_simulation.py
==========================
Phase 3: Time-Optimal Bitmap Index Under Space Constraint

Based on Section 8 of "Bitmap Index Design and Evaluation"
(Chan & Ioannidis, SIGMOD 1998)

Implements:
  - Simulation 3.1: TimeOptAlg  (Exact solver with pruning, Figure 12/14)
  - Simulation 3.2: TimeOptHeur (Heuristic: FindSmallestN + RefineIndex, Figure 15)
  - Simulation 3.3: Table 2 Reproduction (Heuristic Accuracy vs Optimal)
  - Simulation 3.4: Empirical Query Benchmarking (Hot/Cold Cache)
  - Simulation 3.5: Visualizations (5 publication-ready plots)

Directory Layout:
  code/phase3_time_optimal/
      time_optimal_simulation.py   <- this file
      results/                     <- all PNG plots saved here
"""

import csv
import gc
import math
import os
import random
import statistics
import sys
import time as time_mod
from functools import lru_cache

# Setup environment and directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATION_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))
RESULTS_DIR = os.path.join(SIMULATION_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

CODE_ROOT = os.path.normpath(os.path.join(SIMULATION_DIR, '..'))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)
COMMON_DIR = os.path.join(CODE_ROOT, 'common')
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from bitmap_utils import (
    OpCounter, all_ones, build_range_ncomp, decompose_value,
    get_range_leq_bitmap, ground_truth_query,
    benchmark_hot, benchmark_cold, evict_cpu_cache
)
from range_eval_algorithms_section3_1.code.rangeeval_vs_rangeeval_opt_benchmark import rangeeval_opt_query, QUERY_TYPES

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================================
# COST MODEL FORMULAS (Equations 3 & 4)
# ============================================================================

def space_of(bases):
    """Space(I) = sum(b_i - 1) for range encoding (Eq 3)."""
    return sum(b - 1 for b in bases)

def time_of(bases):
    """Time(I) for range encoding (Eq 4).
    Time(I) = 2(n - sum(1/b_i) + (1/3)(1/b_1 - 1))
    In paper convention, components are ordered <b_n, ..., b_1> where b_1
    is chosen to minimize Time(I), i.e., b_1 = max(bases).
    """
    n = len(bases)
    if n == 0:
        return float('inf')
    sum_inv = sum(1.0 / b for b in bases)
    b1 = max(bases)
    return 2.0 * (n - sum_inv + (1.0 / 3.0) * (1.0 / b1 - 1.0))

def product_of(bases):
    p = 1
    for b in bases:
        p *= b
    return p

def time_opt_n(n, C):
    """Theorem 6.1 (3): The base of the n-component time-optimal index is
    <ceil(C / 2^(n-1)), 2, ..., 2>.
    """
    b_high = max(2, math.ceil(C / (2**(n - 1))))
    return [2]*(n - 1) + [b_high]

# ============================================================================
# SIMULATION 3.1: TimeOptAlg — Exact Optimal (Section 8.1, Figure 12)
# ============================================================================
#
# Key insight from the paper:
#   Step 1: Find smallest n such that Space(I_space^n) <= M.
#   Step 2: If the n-component time-optimal (= space-optimal for fixed n) fits, done.
#   Step 3: Find smallest n' > n such that Space(I_time^{n'}) <= M.
#   Step 4: Search only k-component indexes for n < k < n', plus I_time^{n'}.
#
# For *each fixed k*, the search is over all base vectors b_1 <= b_2 <= ... <= b_k
# with sum(b_i - 1) <= M and prod(b_i) >= C.
#
# Since Time(I) = 2(k - sum(1/b_i) + ...) and 1/b_i is a decreasing concave
# function, the optimal base vector for fixed k wants bases as small as possible.
# We use branch-and-bound with strong pruning.

def _space_optimal_base(n, C):
    """Uniform base for n-component space-optimal index: ceil(C^(1/n))."""
    b = max(2, math.ceil(C ** (1.0 / n)))
    return b

def _space_of_space_optimal(n, C):
    b = _space_optimal_base(n, C)
    return n * (b - 1)

def time_opt_alg(M, C):
    """Exact solver — exhaustive search over ALL valid component counts.

    For each k from 1 to max_n, enumerate all k-component base vectors
    with sum(b_i - 1) <= M and prod(b_i) >= C, keeping the global minimum
    Time(I). Uses branch-and-bound with strong pruning.

    Returns (bases, time, space) or (None, inf, None).
    """
    max_n = min(math.ceil(math.log2(C)) + 1, M)

    best_bases = None
    best_time = float('inf')

    for k in range(1, max_n + 1):
        if k > M:
            break
        _out = []
        _search_k_component(k, M, C, best_time, [], 1, 0, 1, _out)
        for bases_cand, t in _out:
            if t < best_time:
                best_time = t
                best_bases = bases_cand

    if best_bases is None:
        return None, float('inf'), None
    return best_bases, best_time, space_of(best_bases)


def time_opt_alg_paper(M, C):
    """Algorithm TimeOptAlg following the EXACT paper specification (Figure 12).

    1) let n be the smallest number such that Space(I_space^n) <= M (via FindSmallestN)
    2) if (Space(I_time^n) <= M) then return I_time^n;
    3) let n' be the smallest number such that n' > n and Space(I_time^{n'}) <= M;
    4) let Z = {I_k : n <= k < n', Space(I_k) <= M} U {I_time^{n'}};
    5) return I' in Z such that Time(I') <= Time(I), for all I in Z;

    This produces the exact theoretical comparison baseline for Table 2.
    """
    max_comp = math.ceil(math.log2(C)) + 1
    n, _ = find_smallest_n(M, C)
    itn = time_opt_n(n, C)
    if space_of(itn) <= M:
        return itn, time_of(itn), space_of(itn)

    # Step 3: smallest n' > n such that Space(I_time^{n'}) <= M
    n_prime = None
    for k in range(n + 1, max_comp + 2):
        itk = time_opt_n(k, C)
        if space_of(itk) <= M:
            n_prime = k
            break
    if n_prime is None:
        n_prime = max_comp + 1

    # Step 4: Z = {I_k : n <= k < n', Space <= M} U {I_time^{n'}}
    best_b, best_t = None, float('inf')
    if n_prime <= max_comp + 1:
        it_np = time_opt_n(n_prime, C)
        if space_of(it_np) <= M:
            best_b = it_np
            best_t = time_of(it_np)

    for k in range(n, n_prime):
        if k > M:
            break
        _out = []
        _search_k_component(k, M, C, best_t, [], 1, 0, 1, _out)
        for bases_cand, t in _out:
            if t < best_t:
                best_t = t
                best_b = bases_cand

    if best_b is None:
        return None, float('inf'), None
    return best_b, best_t, space_of(best_b)


def min_space_to_cover(needed, rr, min_b):
    """Theorem 6.1: Minimum space needed by rr components each >= min_b to cover product needed."""
    if needed <= 1:
        return rr * (min_b - 1)
    if min_b ** rr >= needed:
        return rr * (min_b - 1)
    b = max(min_b, math.ceil(needed ** (1.0 / rr)))
    if b - 1 >= min_b:
        for tau in range(1, rr + 1):
            if (b ** tau) * ((b - 1) ** (rr - tau)) >= needed:
                return tau * (b - 1) + (rr - tau) * (b - 2)
    return rr * (b - 1)


def _search_k_component(k, M, C, prune_time, current, depth, space_used,
                        prod_so_far, results):
    """Branch-and-bound search for k-component bases with strong pruning.

    Bases are assigned in non-decreasing order (canonical) to avoid duplicates.
    """
    remaining = k - depth + 1  # Components left including this one
    min_b = current[-1] if current else 2  # Non-decreasing order

    if remaining == 1:
        # Last component: must have b >= ceil(C / prod_so_far)
        b_min = max(min_b, math.ceil(C / prod_so_far) if prod_so_far > 0 else 2)
        b_max = M - space_used + 1  # b - 1 <= remaining space

        if b_min > b_max:
            return

        for b in range(b_min, min(b_min + 4, b_max + 1)):
            bases = current + [b]
            t = time_of(bases)
            if t < prune_time:
                results.append((bases[:], t))
        return

    # Space left for remaining components (each needs at least 1 = base 2)
    max_space_this = M - space_used - (remaining - 1)
    b_upper = max_space_this + 1  # b - 1 <= max_space_this

    for b in range(min_b, b_upper + 1):
        new_space = space_used + (b - 1)
        new_prod = prod_so_far * b
        leftover_space = M - new_space

        # Pruning 1: Can remaining-1 components cover C/new_prod?
        rr = remaining - 1
        needed = math.ceil(C / new_prod) if new_prod > 0 and C > new_prod else 1
        min_space_for_rest = min_space_to_cover(needed, rr, b)
        if min_space_for_rest > leftover_space:
            continue  # Impossible

        # Pruning 2: Optimistic lower bound on Time
        sum_inv_so_far = sum(1.0 / x for x in current) + 1.0 / b
        optimistic_sum_inv = sum_inv_so_far + rr * (1.0 / b)
        optimistic_b1 = max(b, leftover_space + 1)
        optimistic_time = 2.0 * (k - optimistic_sum_inv +
                                  (1.0 / 3.0) * (1.0 / optimistic_b1 - 1.0))
        if optimistic_time >= prune_time:
            break  # Increasing b only makes it worse; prune entire subtree

        _search_k_component(k, M, C, prune_time, current + [b], depth + 1,
                            new_space, new_prod, results)
        # Update prune_time from results found so far
        if results:
            best_found = min(t for _, t in results)
            if best_found < prune_time:
                prune_time = best_found


# ============================================================================
# SIMULATION 3.2: TimeOptHeur — Heuristic Solver (Section 8.2, Figure 15)
# ============================================================================

def find_smallest_n(M, C):
    """Algorithm FindSmallestN (Figure 15).

    Returns (n, bases) where bases has n-r copies of b and r copies of b+1,
    with sum(b_i - 1) = M (using all space) and prod(b_i) >= C.
    """
    n = 0
    while True:
        n += 1
        b = (M + n) // n       # = floor((M + n) / n)
        r = (M + n) % n        # r components get base b+1, (n-r) get b

        if b < 2:
            # Need b >= 2; with this M and n, b is too small.
            # Won't happen for reasonable inputs, but guard.
            continue

        # Product check: b^(n-r) * (b+1)^r >= C ?
        prod = (b ** (n - r)) * ((b + 1) ** r) if r > 0 else b ** n
        if prod >= C:
            break

    bases = [b] * (n - r) + [b + 1] * r
    return n, sorted(bases)


def refine_index(bases, C):
    """Algorithm RefineIndex (Figure 15).

    Iteratively adjust pairs of base numbers using Theorem 8.1 to improve
    time-efficiency while maintaining Space(I) <= original and prod(b_i) >= C.
    """
    n = len(bases)
    if n <= 1:
        return bases[:]

    seq = sorted(bases[:])
    prod = product_of(seq)
    b_prime = [0] * n

    for i in range(n, 1, -1):
        bp = min(seq)
        seq.remove(bp)
        if bp > 2 and len(seq) > 0:
            bq = min(seq)
            seq.remove(bq)
            other_prod = prod // (bp * bq)
            # Exact Figure 15 formula: (bp + bq)**2 - (4 * bp * bq * C) / prod
            disc = (bp + bq) ** 2 - (4.0 * bp * bq * C) / prod
            if disc >= 0:
                delta = int(math.floor((bp - bq + math.sqrt(disc)) / 2.0))
                if delta > 0:
                    delta = min(delta, bp - 2)
                    while delta > 0 and other_prod * (bp - delta) * (bq + delta) < C:
                        delta -= 1
                    if delta > 0:
                        prod = other_prod * (bp - delta) * (bq + delta)
                        bp = bp - delta
                        bq = bq + delta
            seq.append(bq)
        b_prime[i - 1] = bp

    rest_prod = product_of(b_prime[1:])
    b_prime[0] = max(2, math.ceil(C / rest_prod))
    return sorted(b_prime)


def time_opt_heur(M, C):
    """Algorithm TimeOptHeur (Figure 12 right side).

    1. FindSmallestN(M, C) → (n, I_n)
    2. If Space(I_time^n) <= M, return I_time^n.
    3. Else return RefineIndex(I_n, C).
    """
    n, seed_bases = find_smallest_n(M, C)
    itn = time_opt_n(n, C)
    if space_of(itn) <= M:
        return itn, time_of(itn), space_of(itn)

    refined = refine_index(seed_bases, C)
    return refined, time_of(refined), space_of(refined)


# ============================================================================
# SIMULATION 3.3: Table 2 Reproduction
# ============================================================================

def reproduce_table2(cardinalities=None):
    """Reproduce Table 2: Effectiveness of Heuristic Approach.

    Uses time_opt_alg_paper (paper's restricted search) for fair comparison,
    matching the paper's methodology. Also records global-optimal for analysis.
    """
    if cardinalities is None:
        cardinalities = [50, 100, 200, 500, 1000, 2000]

    table2_results = []

    for C in cardinalities:
        print(f"\n  Table 2: C = {C} ...", end=' ', flush=True)
        t0 = time_mod.perf_counter()

        n_min_comp = math.ceil(math.log2(C))
        M_min = n_min_comp
        M_max = C - 1

        total_M = 0
        optimal_matches = 0
        max_scan_diff = 0.0
        sum_scan_diff = 0.0
        diff_count = 0
        opt_times_list = []
        heur_times_list = []

        for M in range(M_min, M_max + 1):
            # Paper's TimeOptAlg (restricted search)
            opt_bases, opt_time, opt_space = time_opt_alg_paper(M, C)
            if opt_bases is None:
                continue

            heur_bases, heur_time, heur_space = time_opt_heur(M, C)
            if heur_space is not None and heur_space > M:
                continue
            if product_of(heur_bases) < C:
                continue

            total_M += 1
            scan_diff = heur_time - opt_time
            opt_times_list.append(opt_time)
            heur_times_list.append(heur_time)

            if abs(scan_diff) < 1e-9:
                optimal_matches += 1
            else:
                diff_count += 1
                sum_scan_diff += abs(scan_diff)
                max_scan_diff = max(max_scan_diff, abs(scan_diff))

        elapsed = time_mod.perf_counter() - t0
        pct_optimal = (optimal_matches / total_M * 100) if total_M > 0 else 0
        avg_diff = (sum_scan_diff / diff_count) if diff_count > 0 else 0.0

        result = {
            'C': C,
            'total_M': total_M,
            'optimal_matches': optimal_matches,
            'pct_optimal': pct_optimal,
            'max_scan_diff': max_scan_diff,
            'avg_scan_diff': avg_diff,
            'diff_count': diff_count,
            'opt_times': opt_times_list,
            'heur_times': heur_times_list,
            'elapsed_s': elapsed,
        }
        table2_results.append(result)
        print(f"done in {elapsed:.1f}s  |  {pct_optimal:.1f}% optimal  |  "
              f"max diff = {max_scan_diff:.4f}")

    return table2_results


# ============================================================================
# SIMULATION 3.4: Empirical Query Benchmarking
# ============================================================================

def run_empirical_benchmark(C=100, N=100_000, budgets=None):
    """Build indexes for selected (C, M) pairs, run queries, compare latency."""
    if budgets is None:
        n_min = math.ceil(math.log2(C))
        budgets = sorted(set([
            n_min,
            int(C * 0.15),
            int(C * 0.25),
            int(C * 0.40),
            int(C * 0.60),
            C - 1
        ]))

    print(f"\n  Empirical Benchmark: C={C}, N={N:,}")
    print(f"  Space budgets: {budgets}")

    data = [random.randrange(C) for _ in range(N)]
    query_values = sorted(set([int(C * i / 50) for i in range(50)] + [0, C - 1]))
    query_values = [v for v in query_values if 0 <= v < C]

    bench_results = []

    for M in budgets:
        print(f"\n  --- Budget M = {M} ---")

        opt_bases, opt_time_th, opt_space = time_opt_alg(M, C)
        heur_bases, heur_time_th, heur_space = time_opt_heur(M, C)
        if opt_bases is None:
            print(f"    No valid index for M={M}, skipping.")
            continue

        for label, bases, theory_time, sp in [
            ('Optimal', opt_bases, opt_time_th, opt_space),
            ('Heuristic', heur_bases, heur_time_th, heur_space)
        ]:
            print(f"    {label}: bases={bases}, Space={sp}, Time(theory)={theory_time:.3f}")
            idx = build_range_ncomp(data, bases, N)

            # Explicit correctness verification against ground truth across all 6 operators
            verify_sample_vals = [0, C // 4, C // 2, 3 * C // 4, C - 1]
            for v_chk in verify_sample_vals:
                for qt in QUERY_TYPES:
                    actual_bm = rangeeval_opt_query(idx, bases, v_chk, N, qt)
                    expected_bm = ground_truth_query(data, N, qt, v_chk)
                    assert actual_bm == expected_bm, (
                        f"Correctness failure: {label} index with bases={bases}, "
                        f"op='{qt}', value={v_chk}"
                    )
            print(f"      ✓ Correctness verified: rangeeval_opt matches ground truth across all 6 operators")

            # Count ops
            total_ops = OpCounter()
            qcount = 0
            for v in query_values:
                for qt in QUERY_TYPES:
                    ops = OpCounter()
                    rangeeval_opt_query(idx, bases, v, N, qt, ops)
                    total_ops.and_ops += ops.and_ops
                    total_ops.or_ops += ops.or_ops
                    total_ops.not_ops += ops.not_ops
                    total_ops.xor_ops += ops.xor_ops
                    total_ops.scans += ops.scans
                    qcount += 1

            avg_scans = total_ops.scans / qcount
            avg_ops = total_ops.total_ops() / qcount

            # Hot cache
            hot_times = []
            for v in query_values[:20]:
                for qt in ['leq', 'eq']:
                    fn = lambda v=v, qt=qt: rangeeval_opt_query(idx, bases, v, N, qt)
                    t = benchmark_hot(fn, warmup=15, iterations=100, rounds=5)
                    hot_times.append(t)
            hot_med = statistics.median(hot_times) if hot_times else 0

            # Cold cache
            cold_times = []
            for v in random.sample(query_values, min(15, len(query_values))):
                fn = lambda v=v: rangeeval_opt_query(idx, bases, v, N, 'leq')
                t = benchmark_cold(fn, trials=10)
                cold_times.append(t)
            cold_med = statistics.median(cold_times) if cold_times else 0

            bench_results.append({
                'M': M,
                'label': label,
                'bases': bases,
                'space': sp,
                'time_theory': theory_time,
                'avg_scans': avg_scans,
                'avg_ops': avg_ops,
                'hot_median_us': hot_med,
                'cold_median_us': cold_med,
            })
            print(f"      Scans={avg_scans:.2f}  Ops={avg_ops:.1f}  "
                  f"Hot={hot_med:.1f}µs  Cold={cold_med:.1f}µs")

    return bench_results


# ============================================================================
# SIMULATION 3.5: PLOTS
# ============================================================================

CLR = {
    'opt': '#1A5276', 'heur': '#C0392B', 'match': '#27AE60',
    'mismatch': '#E74C3C', 'hot': '#E74C3C', 'cold': '#3498DB',
}


def plot1_heuristic_accuracy(table2):
    """Plot 1: Heuristic Accuracy vs Cardinality (Table 2 bar chart)."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    C_vals = [r['C'] for r in table2]
    pcts = [r['pct_optimal'] for r in table2]
    diffs = [r['max_scan_diff'] for r in table2]
    x = np.arange(len(C_vals))

    bars = ax1.bar(x, pcts, 0.45, color=CLR['match'], alpha=0.85,
                   edgecolor='black', linewidth=0.5, label='% Optimal Match')
    for bar, p in zip(bars, pcts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f'{p:.1f}%', ha='center', fontsize=10, fontweight='bold')

    ax1.set_xlabel('Attribute Cardinality (C)', fontsize=12)
    ax1.set_ylabel('% Heuristic = Optimal', fontsize=12, color=CLR['match'])
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(c) for c in C_vals], fontsize=11)
    ax1.set_ylim(0, 115)

    ax2 = ax1.twinx()
    ax2.plot(x, diffs, 's--', color=CLR['mismatch'], markersize=10, linewidth=2,
             label='Max Scan Difference')
    ax2.set_ylabel('Max Scan Difference', fontsize=12, color=CLR['mismatch'])
    ax2.set_ylim(bottom=0)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='lower right', fontsize=10)
    ax1.set_title('Table 2 Reproduction: Heuristic Accuracy vs Cardinality\n'
                  '(TimeOptHeur vs TimeOptAlg across all valid space constraints)',
                  fontsize=13, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'figure13_heuristic_accuracy_and_scan_difference.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Plot 1 → {path}")


def plot2_time_vs_space_budget(C_list=None):
    """Plot 2: Time-Space Budget Curve (Time(I) vs space budget M)."""
    if C_list is None:
        C_list = [100, 500, 1000]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = ['#1A5276', '#E67E22', '#C0392B']
    markers = ['o', 's', '^']

    for i, C in enumerate(C_list):
        M_min = math.ceil(math.log2(C))
        M_max = C - 1
        step = max(1, (M_max - M_min) // 60)
        sample = list(range(M_min, M_max + 1, step))
        if M_max not in sample:
            sample.append(M_max)

        Ms, ot, ht = [], [], []
        for M in sample:
            ob, otv, _ = time_opt_alg(M, C)
            hb, htv, _ = time_opt_heur(M, C)
            if ob:
                Ms.append(M)
                ot.append(otv)
                ht.append(htv)

        every = max(1, len(Ms) // 12)
        ax.plot(Ms, ot, '-', color=colors[i], linewidth=2,
                marker=markers[i], markevery=every, markersize=5,
                label=f'Optimal (C={C})')
        ax.plot(Ms, ht, '--', color=colors[i], linewidth=1.2, alpha=0.7,
                label=f'Heuristic (C={C})')

    ax.set_xlabel('Space Budget M (max bitmaps)', fontsize=12)
    ax.set_ylabel('Time(I) — Expected Bitmap Scans', fontsize=12)
    ax.set_title('Time-Space Budget Curve: Query Cost vs Space Budget\n'
                 '(Section 8: Time-Optimal Index Under Space Constraint)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.35)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'figure14_query_time_vs_space_budget_tradeoff.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Plot 2 → {path}")


def plot3_opt_vs_heur_scatter(table2):
    """Plot 3: Optimal vs Heuristic scatter for C=100 and C=1000."""
    targets = [r for r in table2 if r['C'] in [100, 1000]]
    if len(targets) < 2:
        targets = table2[-2:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, res in zip(axes, targets):
        C = res['C']
        ot = res['opt_times']
        ht = res['heur_times']
        if not ot:
            continue

        m_o = [o for o, h in zip(ot, ht) if abs(o - h) < 1e-9]
        m_h = [h for o, h in zip(ot, ht) if abs(o - h) < 1e-9]
        d_o = [o for o, h in zip(ot, ht) if abs(o - h) >= 1e-9]
        d_h = [h for o, h in zip(ot, ht) if abs(o - h) >= 1e-9]

        vals = ot + ht
        lo, hi = min(vals) * 0.95, max(vals) * 1.05
        ax.plot([lo, hi], [lo, hi], '--', color='gray', linewidth=1, alpha=0.6,
                label='y = x (perfect match)')
        ax.scatter(m_o, m_h, color=CLR['match'], s=20, alpha=0.5,
                   label=f'Match ({len(m_o)})')
        if d_o:
            ax.scatter(d_o, d_h, color=CLR['mismatch'], s=35, marker='x',
                       alpha=0.8, label=f'Differ ({len(d_o)})')
        ax.set_xlabel('Optimal Time (scans)', fontsize=11)
        ax.set_ylabel('Heuristic Time (scans)', fontsize=11)
        ax.set_title(f'C = {C}  ({res["pct_optimal"]:.1f}% match)',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')

    fig.suptitle('Optimal vs Heuristic Scatter (above diagonal = suboptimal heuristic)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'optimal_vs_heuristic_cost_scatter.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Plot 3 → {path}")


def plot4_solver_scalability():
    """Plot 4: Solver Execution Scalability (log-scale runtime comparison)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    test_C = [20, 50, 100, 200, 500, 1000, 2000]
    opt_ms, heur_ms = [], []

    for C in test_C:
        M = max(math.ceil(math.log2(C)), int(C * 0.3))

        # Warmup and repeated runs to eliminate timing jitter
        for _ in range(5):
            time_opt_alg(M, C)

        n_runs = 30 if C <= 500 else 15
        opt_samples = []
        for _ in range(n_runs):
            t0 = time_mod.perf_counter()
            time_opt_alg(M, C)
            opt_samples.append((time_mod.perf_counter() - t0) * 1000.0)
        t_opt = statistics.median(opt_samples)

        for _ in range(10):
            time_opt_heur(M, C)

        t0 = time_mod.perf_counter()
        n_heur_runs = 1000
        for _ in range(n_heur_runs):
            time_opt_heur(M, C)
        t_heur_per = (time_mod.perf_counter() - t0) * 1000.0 / n_heur_runs  # ms per run

        opt_ms.append(t_opt)
        heur_ms.append(t_heur_per)
        print(f"    C={C:>5d}  M={M:>4d}  Opt={t_opt:>10.3f}ms  Heur={t_heur_per:>8.4f}ms")

    ax.plot(test_C, opt_ms, 'o-', color=CLR['opt'], linewidth=2.5, markersize=8,
            label='TimeOptAlg (Exact Branch-and-Bound)')
    ax.plot(test_C, heur_ms, 's--', color=CLR['heur'], linewidth=2.5, markersize=8,
            label='TimeOptHeur (Heuristic)')
    ax.set_yscale('log')
    ax.set_xlabel('Attribute Cardinality (C)', fontsize=12)
    ax.set_ylabel('Solver Execution Time (ms, log scale)', fontsize=12)
    ax.set_title('Solver Scalability: Exact Branch-and-Bound vs Heuristic\n'
                 '(Space budget M = 0.3 × C)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)

    for i, C in enumerate(test_C):
        if heur_ms[i] > 0:
            sp = opt_ms[i] / heur_ms[i]
            if sp > 5:
                ax.annotate(f'{sp:.0f}×', (C, opt_ms[i]),
                            textcoords='offset points', xytext=(10, 5),
                            fontsize=9, color=CLR['opt'], fontweight='bold')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'branch_and_bound_vs_heuristic_scalability.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Plot 4 → {path}")


def plot5_empirical_vs_theory(bench):
    """Plot 5: Empirical Latency vs Theoretical Scans."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, key, title in [
        (axes[0], 'hot_median_us', 'Hot Cache Latency (µs)'),
        (axes[1], 'cold_median_us', 'Cold Cache Latency (µs)'),
    ]:
        tt = [r['time_theory'] for r in bench]
        et = [r[key] for r in bench]
        ll = [r['label'] for r in bench]

        for lbl, clr, mk in [('Optimal', CLR['opt'], 'o'),
                               ('Heuristic', CLR['heur'], 's')]:
            tx = [t for t, l in zip(tt, ll) if l == lbl]
            ex = [e for e, l in zip(et, ll) if l == lbl]
            if tx:
                ax.scatter(tx, ex, color=clr, s=70, marker=mk,
                           edgecolors='black', linewidth=0.5, label=lbl, zorder=5)

        if len(tt) >= 2:
            c = np.polyfit(tt, et, 1)
            fx = np.linspace(min(tt), max(tt), 100)
            r2 = np.corrcoef(tt, et)[0, 1] ** 2
            ax.plot(fx, np.polyval(c, fx), '--', color='gray', linewidth=1.5,
                    alpha=0.7, label=f'Linear fit (R²={r2:.3f})')

        ax.set_xlabel('Theoretical Scans (Time(I))', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title.split(' (')[0], fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)

    fig.suptitle('Empirical Validation: Query Latency vs Theoretical Scans\n'
                 '(C=100, N=100,000)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'physical_query_latency_vs_theoretical_cost.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Plot 5 → {path}")


# ============================================================================
# TABLE 2 CONSOLE & CSV
# ============================================================================

def print_table2(table2):
    print(f"\n{'='*78}")
    print("TABLE 2: Effectiveness of Heuristic Approach to Select")
    print("         Time-Optimal Bitmap Index Under Space Constraint")
    print(f"{'='*78}")
    print(f"{'C':>6s}  {'% Optimal':>10s}  {'Max Diff':>10s}  "
          f"{'Avg Diff':>10s}  {'M tested':>8s}  {'Time(s)':>8s}")
    print(f"{'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    for r in table2:
        print(f"{r['C']:>6d}  {r['pct_optimal']:>9.1f}%  "
              f"{r['max_scan_diff']:>10.4f}  {r['avg_scan_diff']:>10.6f}  "
              f"{r['total_M']:>8d}  {r['elapsed_s']:>7.1f}s")
    print(f"{'='*78}\n")


def export_table2_csv(table2):
    path = os.path.join(RESULTS_DIR, 'table2_optimal_vs_heuristic_reproduction.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['C', 'pct_optimal', 'max_scan_diff', 'avg_scan_diff',
                     'total_M', 'matches', 'differences'])
        for r in table2:
            w.writerow([r['C'], f"{r['pct_optimal']:.2f}",
                        f"{r['max_scan_diff']:.6f}", f"{r['avg_scan_diff']:.8f}",
                        r['total_M'], r['optimal_matches'], r['diff_count']])
    print(f"  CSV → {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t_start = time_mod.perf_counter()

    print("=" * 78)
    print("  PHASE 3: Time-Optimal Bitmap Index Under Space Constraint")
    print("  Based on Section 8 of Chan & Ioannidis (SIGMOD 1998)")
    print("=" * 78)

    # --- SIM 3.1 & 3.2: Quick verification ---
    print("\n" + "-" * 78)
    print("  SIM 3.1 & 3.2: Algorithm Verification")
    print("-" * 78)

    for C, M in [(1000, 59), (100, 15), (50, 10)]:
        ob, ot, os_ = time_opt_alg(M, C)
        hb, ht, hs = time_opt_heur(M, C)
        print(f"\n  C={C}, M={M}")
        print(f"    Optimal  : bases={ob}, Space={os_}, Time={ot:.4f}")
        print(f"    Heuristic: bases={hb}, Space={hs}, Time={ht:.4f}")
        match = "✓ MATCH" if abs(ot - ht) < 1e-9 else f"✗ diff={ht-ot:.4f}"
        print(f"    Result   : {match}")

    # --- SIM 3.3: Table 2 ---
    print("\n" + "-" * 78)
    print("  SIM 3.3: Table 2 Reproduction")
    print("-" * 78)
    table2 = reproduce_table2([50, 100, 200, 500, 1000, 2000])
    print_table2(table2)
    export_table2_csv(table2)

    # --- SIM 3.5: Plots 1-4 ---
    print("\n" + "-" * 78)
    print("  SIM 3.5: Generating Plots 1-4")
    print("-" * 78)
    plot1_heuristic_accuracy(table2)
    plot2_time_vs_space_budget([100, 500, 1000])
    plot3_opt_vs_heur_scatter(table2)

    print("\n  Plot 4: Solver Scalability ...")
    plot4_solver_scalability()

    # --- SIM 3.4: Empirical ---
    print("\n" + "-" * 78)
    print("  SIM 3.4: Empirical Query Benchmarking")
    print("-" * 78)
    bench = run_empirical_benchmark(C=100, N=100_000)

    print("\n  Plot 5: Empirical vs Theory ...")
    plot5_empirical_vs_theory(bench)

    # --- DONE ---
    elapsed = time_mod.perf_counter() - t_start
    print("\n" + "=" * 78)
    print(f"  PHASE 3 COMPLETE — Total runtime: {elapsed:.1f}s")
    print(f"  Results: {RESULTS_DIR}")
    print("=" * 78)
    print("\n  Generated files:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        fp = os.path.join(RESULTS_DIR, f)
        print(f"    {f:45s}  {os.path.getsize(fp):>8,} bytes")


if __name__ == '__main__':
    main()
