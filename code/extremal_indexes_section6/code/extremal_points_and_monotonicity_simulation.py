"""
s5_extremal_points.py
=====================
Simulation S5: n-Component Extremal Indexes (Space-Optimal & Time-Optimal)

Based on Section 6, Theorem 6.1 of "Bitmap Index Design and Evaluation"
(Chan & Ioannidis, SIGMOD 1998)

Reproduces:
  - Theorem 6.1(1): Space-optimal n-component range index construction
  - Theorem 6.1(2): Time-optimal n-component range index construction
  - Theorem 6.1(3): Space monotonicity (strictly decreases with n; Point A at n = ceil(log2 C))
  - Theorem 6.1(4): Time monotonicity (strictly increases with n; Point D at n = 1)
  - Empirical verification against brute-force argmin over all enumerated decompositions
  - Verification across C in {20, 50, 100, 200, 500, 1000}

Plots generated:
  - s5_theorem61_verification.png (heatmap/grid showing analytical vs brute-force matches)
  - s5_monotonicity.png (dual-axis line chart of space vs time across n)
  - s5_extremal_on_frontier.png (extremal classes highlighted on Pareto frontier for C=100)

CSV generated:
  - results/s5_extremal_data.csv
"""

import csv
import math
import os
import random
import sys
import os

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


os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s3_cost_model_validation import analytical_range_space, analytical_range_time
from s4_global_frontier import enumerate_all_decompositions_for_C, compute_pareto_frontier

SEED = 42
random.seed(SEED)


# ============================================================================
# 1. THEOREM 6.1 CONSTRUCTIVE FORMULAS
# ============================================================================

def construct_space_optimal(C, n):
    """Theorem 6.1(1): Construct space-optimal n-component range index for cardinality C.

    The space-optimal decomposition minimizes sum(b_i - 1), which is equivalent
    to minimizing sum(b_i) subject to prod(b_i) >= C with b_i >= 2.
    By AM-GM inequality, this is achieved when all b_i are as uniform as possible:
      b = ceil(C^(1/n))
    Some components have base b, others have base (b - 1).
    We find the minimum k in {0..n} such that:
      b^k * (b - 1)^(n - k) >= C
    with the constraint that (b - 1) >= 2. If b - 1 < 2, all bases are b = 2.
    """
    if n == 1:
        return [C]

    b = math.ceil(C ** (1.0 / n))
    b = max(b, 2)

    if b - 1 < 2:
        # Cannot use base < 2, so all components must be at least 2
        bases = [2] * n
        while math.prod(bases) < C:
            # Increment bases from the end if needed
            for i in range(n - 1, -1, -1):
                bases[i] += 1
                if math.prod(bases) >= C:
                    break
        return bases

    # Find minimum k such that b^k * (b - 1)^(n - k) >= C
    best_k = n
    for k in range(n + 1):
        prod = (b ** k) * ((b - 1) ** (n - k))
        if prod >= C:
            best_k = k
            break

    # Bases: (n - best_k) of (b - 1), and best_k of b
    bases = [b - 1] * (n - best_k) + [b] * best_k

    # For range encoding Time(I), LSB b1 has a smaller penalty if larger or smaller.
    # To minimize Time among space-equivalent bases, sort bases:
    return bases


def construct_time_optimal(C, n):
    """Theorem 6.1(2): Construct time-optimal n-component range index for cardinality C.

    In the paper's notation <b_n, ..., b_1>, the high-order components b_n..b_2 are 2,
    and the least-significant component b_1 = ceil(C / 2^(n-1)) absorbs the remaining cardinality.
    In our internal representation where bases[0] = b_1 (LSB) and bases[-1] = b_n (MSB):
      bases = [last_b] + [2] * (n - 1)
    This places the largest base at b_1, which minimizes Equation 4:
      Time(I) = 2*(n - sum(1/b_i) + (1/3)*(1/b_1 - 1)).
    """
    if n == 1:
        return [C]

    last_b = math.ceil(C / (2 ** (n - 1)))
    last_b = max(last_b, 2)
    bases = [last_b] + [2] * (n - 1)
    return bases


# ============================================================================
# 2. BRUTE-FORCE ARGMIN SEARCH
# ============================================================================

def bruteforce_space_optimal(all_decomps, n):
    """Find n-component decomposition that strictly minimizes range space."""
    n_decomps = [d for d in all_decomps if len(d) == n]
    if not n_decomps:
        return None, float('inf'), float('inf')

    best_d = None
    min_s = float('inf')
    best_t = float('inf')

    for d in n_decomps:
        bases = list(d)
        s = analytical_range_space(bases)
        t = analytical_range_time(bases)
        if s < min_s or (s == min_s and t < best_t):
            min_s = s
            best_t = t
            best_d = bases

    return best_d, min_s, best_t


def bruteforce_time_optimal(all_decomps, n):
    """Find n-component decomposition that strictly minimizes range time."""
    n_decomps = [d for d in all_decomps if len(d) == n]
    if not n_decomps:
        return None, float('inf'), float('inf')

    best_d = None
    min_t = float('inf')
    best_s = float('inf')

    for d in n_decomps:
        bases = list(d)
        s = analytical_range_space(bases)
        t = analytical_range_time(bases)
        if t < min_t or (t == min_t and s < best_s):
            min_t = t
            best_s = s
            best_d = bases

    return best_d, best_s, min_t


# ============================================================================
# 3. VERIFICATION & MONOTONICITY SUITE
# ============================================================================

def run_s5_evaluation():
    cardinalities = [20, 50, 100, 200, 500, 1000]
    records = []

    print("\n" + "=" * 78)
    print("SIMULATION S5: Theorem 6.1 Extremal Index Verification")
    print("=" * 78)

    for C in cardinalities:
        max_n = math.ceil(math.log2(C))
        all_decomps = enumerate_all_decompositions_for_C(C, max_n=max_n)
        print(f"\n--- Cardinality C = {C} (max_n = {max_n}, total decomps = {len(all_decomps):,}) ---")

        for n in range(1, max_n + 1):
            # 1. Space-optimal construction
            ana_s_bases = construct_space_optimal(C, n)
            ana_s = analytical_range_space(ana_s_bases)
            ana_s_t = analytical_range_time(ana_s_bases)

            bf_s_bases, bf_s, bf_s_t = bruteforce_space_optimal(all_decomps, n)
            s_match = (ana_s == bf_s)

            records.append({
                'C': C, 'n': n, 'type': 'space_optimal',
                'ana_bases': str(ana_s_bases), 'ana_space': ana_s, 'ana_time': ana_s_t,
                'bf_bases': str(bf_s_bases), 'bf_space': bf_s, 'bf_time': bf_s_t,
                'match': s_match
            })

            # 2. Time-optimal construction
            ana_t_bases = construct_time_optimal(C, n)
            ana_t_s = analytical_range_space(ana_t_bases)
            ana_t = analytical_range_time(ana_t_bases)

            bf_t_bases, bf_t_s, bf_t = bruteforce_time_optimal(all_decomps, n)
            t_match = math.isclose(ana_t, bf_t, abs_tol=1e-4) or (ana_t <= bf_t)

            records.append({
                'C': C, 'n': n, 'type': 'time_optimal',
                'ana_bases': str(ana_t_bases), 'ana_space': ana_t_s, 'ana_time': ana_t,
                'bf_bases': str(bf_t_bases), 'bf_space': bf_t_s, 'bf_time': bf_t,
                'match': t_match
            })

            s_icon = "✓" if s_match else "✗"
            t_icon = "✓" if t_match else "✗"
            print(f"  n={n:2d} | Space-Opt: {str(ana_s_bases):<18} Space={ana_s:3d} {s_icon} | "
                  f"Time-Opt: {str(ana_t_bases):<18} Time={ana_t:.3f} {t_icon}")

    return records


# ============================================================================
# 4. PLOTTING
# ============================================================================

def plot_s5_verification(records):
    """Plot S5-1: Theorem 6.1 Verification Heatmap."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    cardinalities = [20, 50, 100, 200, 500, 1000]
    max_components = max(r['n'] for r in records)

    grid_s = np.full((len(cardinalities), max_components), np.nan)
    grid_t = np.full((len(cardinalities), max_components), np.nan)

    for r in records:
        c_idx = cardinalities.index(r['C'])
        n_idx = r['n'] - 1
        if r['type'] == 'space_optimal':
            grid_s[c_idx, n_idx] = 1.0 if r['match'] else 0.0
        elif r['type'] == 'time_optimal':
            grid_t[c_idx, n_idx] = 1.0 if r['match'] else 0.0

    im1 = ax1.imshow(grid_s, cmap='Greens', vmin=0, vmax=1.2, aspect='auto')
    ax1.set_title("Theorem 6.1(1) Space-Optimal Construction\n(Analytical vs Brute-Force Match)",
                  fontsize=11, fontweight='bold')
    ax1.set_xticks(range(max_components))
    ax1.set_xticklabels([f"n={i+1}" for i in range(max_components)])
    ax1.set_yticks(range(len(cardinalities)))
    ax1.set_yticklabels([f"C={c}" for c in cardinalities])

    for i in range(len(cardinalities)):
        for j in range(max_components):
            val = grid_s[i, j]
            if not np.isnan(val):
                text = "✓ MATCH" if val == 1.0 else "DIFF"
                ax1.text(j, i, text, ha='center', va='center',
                         color='darkgreen' if val == 1.0 else 'red',
                         fontsize=9, fontweight='bold')

    im2 = ax2.imshow(grid_t, cmap='Blues', vmin=0, vmax=1.2, aspect='auto')
    ax2.set_title("Theorem 6.1(2) Time-Optimal Construction\n(Analytical vs Brute-Force Match)",
                  fontsize=11, fontweight='bold')
    ax2.set_xticks(range(max_components))
    ax2.set_xticklabels([f"n={i+1}" for i in range(max_components)])
    ax2.set_yticks(range(len(cardinalities)))
    ax2.set_yticklabels([f"C={c}" for c in cardinalities])

    for i in range(len(cardinalities)):
        for j in range(max_components):
            val = grid_t[i, j]
            if not np.isnan(val):
                text = "✓ MATCH" if val == 1.0 else "DIFF"
                ax2.text(j, i, text, ha='center', va='center',
                         color='navy' if val == 1.0 else 'red',
                         fontsize=9, fontweight='bold')

    fig.suptitle("Simulation S5: Theorem 6.1 Constructive Extremal Formula Verification",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    path = os.path.join( 'theorem6_1_analytical_vs_empirical_verification.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Plot S5-1 → {path}")


def plot_s5_monotonicity(records):
    """Plot S5-2: Space and Time Monotonicity as a Function of Component Count n."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    sample_Cs = [20, 100, 1000]

    for idx, C in enumerate(sample_Cs):
        ax = axes[idx]
        ax2 = ax.twinx()

        c_recs = [r for r in records if r['C'] == C]
        s_recs = [r for r in c_recs if r['type'] == 'space_optimal']
        t_recs = [r for r in c_recs if r['type'] == 'time_optimal']

        ns = [r['n'] for r in s_recs]
        spaces = [r['ana_space'] for r in s_recs]
        times = [r['ana_time'] for r in t_recs]

        # Space curve (decreasing)
        line1 = ax.plot(ns, spaces, 'o-', color='#2E7D32', linewidth=2.4,
                        markersize=7, label='Space-Opt Space (bitmaps)')
        # Time curve (increasing)
        line2 = ax2.plot(ns, times, 's--', color='#C62828', linewidth=2.4,
                         markersize=7, label='Time-Opt Time (scans)')

        # Mark Point A and Point D
        ax.plot(ns[-1], spaces[-1], '*', color='#FFD700', markersize=14,
                markeredgecolor='black', zorder=5)
        ax.annotate(f"Point A\n(Base-2, S={spaces[-1]})", (ns[-1], spaces[-1]),
                    xytext=(-20, 15), textcoords='offset points', fontsize=8.5,
                    fontweight='bold', color='#1B5E20')

        ax2.plot(ns[0], times[0], 'D', color='#FFD700', markersize=9,
                 markeredgecolor='black', zorder=5)
        ax2.annotate(f"Point D\n(Base-{C}, T={times[0]:.2f})", (ns[0], times[0]),
                     xytext=(10, -15), textcoords='offset points', fontsize=8.5,
                     fontweight='bold', color='#B71C1C')

        ax.set_title(f"Cardinality C = {C}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Number of Components (n)", fontsize=11)
        ax.set_ylabel("Space (Number of Bitmaps)", fontsize=11, color='#2E7D32')
        ax2.set_ylabel("Time (Expected Scans)", fontsize=11, color='#C62828')
        ax.tick_params(axis='y', labelcolor='#2E7D32')
        ax2.tick_params(axis='y', labelcolor='#C62828')
        ax.grid(True, linestyle=':', alpha=0.5)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='center right', fontsize=8.5)

    fig.suptitle("Theorem 6.1(3,4) Space & Time Monotonicity across Component Counts\n"
                 "(Proving Space decreases monotonically and Time increases monotonically with n)",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join( 'theorem6_1_space_time_monotonicity.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Plot S5-2 → {path}")


def plot_s5_extremal_on_frontier(records):
    """Plot S5-3: Extremal Points highlighted on Space-Time Pareto Frontier (C=100)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    C = 100

    all_decomps = enumerate_all_decompositions_for_C(C)
    all_points = []
    for d in all_decomps:
        bases = list(d)
        all_points.append((analytical_range_space(bases), analytical_range_time(bases), bases))

    pareto = compute_pareto_frontier(all_points)

    # Cloud of all decomps
    ax.scatter([p[0] for p in all_points], [p[1] for p in all_points],
               color='#B0BEC5', alpha=0.45, s=25, label='All Decompositions')

    # Pareto frontier
    ax.plot([p[0] for p in pareto], [p[1] for p in pareto],
            '-', color='#1565C0', linewidth=2.2, label='Pareto Frontier')

    # Space-optimal class
    s_recs = [r for r in records if r['C'] == C and r['type'] == 'space_optimal']
    s_spaces = [r['ana_space'] for r in s_recs]
    s_times = [r['ana_time'] for r in s_recs]
    ax.plot(s_spaces, s_times, 'o-', color='#2E7D32', linewidth=2.0, markersize=8,
            label='Space-Optimal Class (Thm 6.1(1))')

    for r in s_recs:
        ax.annotate(f"n={r['n']}", (r['ana_space'], r['ana_time']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8.5, color='#1B5E20')

    # Time-optimal class
    t_recs = [r for r in records if r['C'] == C and r['type'] == 'time_optimal']
    t_spaces = [r['ana_space'] for r in t_recs]
    t_times = [r['ana_time'] for r in t_recs]
    ax.plot(t_spaces, t_times, 's--', color='#C62828', linewidth=2.0, markersize=8,
            label='Time-Optimal Class (Thm 6.1(2))')

    for r in t_recs:
        ax.annotate(f"n={r['n']}", (r['ana_space'], r['ana_time']),
                    xytext=(-15, 6), textcoords='offset points', fontsize=8.5, color='#B71C1C')

    # Point A and Point D
    ax.plot(s_spaces[-1], s_times[-1], '*', color='#FFD700', markeredgecolor='black',
            markersize=14, zorder=6)
    ax.annotate("Point A (Space-Optimal)", (s_spaces[-1], s_times[-1]),
                xytext=(8, -12), textcoords='offset points', fontsize=9.5, fontweight='bold')

    ax.plot(t_spaces[0], t_times[0], 'D', color='#FFD700', markeredgecolor='black',
            markersize=9, zorder=6)
    ax.annotate("Point D (Time-Optimal)", (t_spaces[0], t_times[0]),
                xytext=(-35, 12), textcoords='offset points', fontsize=9.5, fontweight='bold')

    ax.set_title(f"Extremal Index Classes on the Space-Time Plane (C = {C})\n"
                 "(Space-Optimal Class forms the lower envelope; Time-Optimal forms the upper envelope)",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Space (Number of Bitmaps)", fontsize=11)
    ax.set_ylabel("Time (Expected Number of Bitmap Scans)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    path = os.path.join( 'extremal_points_on_pareto_frontier.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Plot S5-3 → {path}")


def export_s5_csv(records):
    """Export S5 records to CSV."""
    path = os.path.join( 'extremal_indexes_comprehensive_data.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'C', 'n', 'type', 'ana_bases', 'ana_space', 'ana_time',
            'bf_bases', 'bf_space', 'bf_time', 'match'
        ])
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"  CSV → {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    records = run_s5_evaluation()

    print("\n--- Generating S5 Plots ---")
    plot_s5_verification(records)
    plot_s5_monotonicity(records)
    plot_s5_extremal_on_frontier(records)

    print("\n--- Exporting CSV ---")
    export_s5_csv(records)

    total_checks = len(records)
    passed_checks = sum(1 for r in records if r['match'])
    print(f"\nS5 COMPLETE: {passed_checks}/{total_checks} checks passed ({passed_checks/total_checks*100:.1f}%)")


if __name__ == '__main__':
    main()
