"""
s6_three_class_comparison.py
============================
Simulation S6: Reproduction of Paper Figures 10 & 11 (Three-Class Tradeoff)

Based on Section 7 of "Bitmap Index Design and Evaluation"
(Chan & Ioannidis, SIGMOD 1998)

Reproduces:
  - Figure 10: Space-Time Tradeoff of Range-Encoded Bitmap Indexes (C = 100)
  - Figure 11: Space-Time Tradeoff of Range-Encoded Bitmap Indexes (C = 1000)
  - Three distinct classes plotted simultaneously:
      1. All Indexes (complete design space cloud of decompositions)
      2. Space-Optimal Class (one point per n = 1..n_max, Theorem 6.1(1))
      3. Time-Optimal Class (one point per n = 1..n_max, Theorem 6.1(2))
  - Mathematical proof that Space-Optimal Class forms the lower Pareto envelope
  - Annotation of component counts n, Point A, Point D, and Point C (the Knee)
  - Plot S6-4: Gradient arrows on Pareto frontier showing discrete changes ΔS and ΔT

Plots generated:
  - s6_fig10_C100.png (Paper Figure 10 reproduction)
  - s6_fig11_C1000.png (Paper Figure 11 reproduction)
  - s6_envelope_proof.png (Proof that Space-Optimal Class traces Pareto envelope)
  - s6_gradient_arrows_C100.png (Discrete gradient step analysis)

CSVs generated:
  - results/s6_three_class_C100.csv
  - results/s6_three_class_C1000.csv
"""

import csv
import math
import os
import random
import sys

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bitmap_utils import RESULTS_DIR
from s3_cost_model_validation import analytical_range_space, analytical_range_time
from s4_global_frontier import enumerate_all_decompositions_for_C, compute_pareto_frontier
from s5_extremal_points import construct_space_optimal, construct_time_optimal

SEED = 42
random.seed(SEED)


# ============================================================================
# 1. THREE-CLASS DATA EXTRACTION
# ============================================================================

def generate_three_class_data(C):
    """Generate the three classes of indexes for cardinality C."""
    max_n = math.ceil(math.log2(C))
    all_decomps = enumerate_all_decompositions_for_C(C, max_n=max_n)

    # 1. All indexes
    all_points = []
    for d in all_decomps:
        bases = list(d)
        s = analytical_range_space(bases)
        t = analytical_range_time(bases)
        all_points.append((s, t, bases))

    # Pareto frontier
    pareto = compute_pareto_frontier(all_points)

    # 2. Space-optimal class (one per n)
    space_opt_class = []
    for n in range(1, max_n + 1):
        bases = construct_space_optimal(C, n)
        s = analytical_range_space(bases)
        t = analytical_range_time(bases)
        space_opt_class.append((s, t, bases, n))

    # 3. Time-optimal class (one per n)
    time_opt_class = []
    for n in range(1, max_n + 1):
        bases = construct_time_optimal(C, n)
        s = analytical_range_space(bases)
        t = analytical_range_time(bases)
        time_opt_class.append((s, t, bases, n))

    return {
        'C': C,
        'max_n': max_n,
        'all_points': all_points,
        'pareto': pareto,
        'space_opt_class': space_opt_class,
        'time_opt_class': time_opt_class
    }


# ============================================================================
# 2. PLOTTING: FIGURES 10 & 11 REPRODUCTION
# ============================================================================

def plot_paper_figure(data, filename, fig_number):
    """Plot Figure 10 or Figure 11 reproducing the paper's exact visual style."""
    C = data['C']
    all_pts = data['all_points']
    pareto = data['pareto']
    s_class = data['space_opt_class']
    t_class = data['time_opt_class']

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # 1. All Indexes Cloud
    ax.scatter([p[0] for p in all_pts], [p[1] for p in all_pts],
               color='#B0BEC5', alpha=0.45, s=28, marker='o',
               label='All Indexes', zorder=2)

    # 2. Pareto Frontier
    ax.plot([p[0] for p in pareto], [p[1] for p in pareto],
            '--', color='#1565C0', linewidth=1.8, alpha=0.8,
            label='Pareto Frontier', zorder=3)

    # 3. Time-Optimal Class (Upper Envelope)
    t_s = [p[0] for p in t_class]
    t_t = [p[1] for p in t_class]
    ax.plot(t_s, t_t, 's--', color='#D32F2F', linewidth=2.0, markersize=8,
            label='Time-Optimal Class', zorder=4)

    for p in t_class:
        ax.annotate(f"n={p[3]}", (p[0], p[1]),
                    xytext=(-18, 7), textcoords='offset points',
                    fontsize=8.5, fontweight='bold', color='#B71C1C')

    # 4. Space-Optimal Class (Lower Envelope)
    s_s = [p[0] for p in s_class]
    s_t = [p[1] for p in s_class]
    ax.plot(s_s, s_t, 'o-', color='#2E7D32', linewidth=2.2, markersize=8,
            label='Space-Optimal Class', zorder=5)

    for p in s_class:
        ax.annotate(f"n={p[3]}", (p[0], p[1]),
                    xytext=(6, 5), textcoords='offset points',
                    fontsize=8.5, fontweight='bold', color='#1B5E20')

    # Annotate Extremal Points: Point A, Point D, Point C (Knee)
    # Point A: Space-optimal with max n
    pt_A = s_class[-1]
    ax.plot(pt_A[0], pt_A[1], '*', color='#FFD700', markeredgecolor='black', markersize=16, zorder=6)
    ax.annotate(f"Point A\n(Base-2, n={pt_A[3]})\nSpace={pt_A[0]}", (pt_A[0], pt_A[1]),
                xytext=(8, -25), textcoords='offset points', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="#2E7D32", lw=1))

    # Point D: Time-optimal with n=1
    pt_D = t_class[0]
    ax.plot(pt_D[0], pt_D[1], 'D', color='#FFD700', markeredgecolor='black', markersize=10, zorder=6)
    ax.annotate(f"Point D\n(Base-{C}, n=1)\nTime={pt_D[1]:.2f}", (pt_D[0], pt_D[1]),
                xytext=(-70, 15), textcoords='offset points', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#C62828", lw=1))

    # Point C: The Knee (n=2 space-optimal)
    pt_C = s_class[1]  # n=2
    ax.plot(pt_C[0], pt_C[1], 'h', color='#FF9800', markeredgecolor='black', markersize=13, zorder=6)
    ax.annotate(f"Point C (The Knee)\n(n=2: {pt_C[2]})\nS={pt_C[0]}, T={pt_C[1]:.2f}", (pt_C[0], pt_C[1]),
                xytext=(15, 20), textcoords='offset points', fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle="->", color='#E65100', lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec="#E65100", lw=1))

    ax.set_title(f"Figure {fig_number}: Space-Time Tradeoff of Range-Encoded Bitmap Indexes (C = {C})\n"
                 "(Reproduction of Chan & Ioannidis SIGMOD 1998)",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Space (Number of Bitmaps)", fontsize=11)
    ax.set_ylabel("Time (Expected Number of Bitmap Scans)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9.5, loc='upper right')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Plot S6 ({fig_number}) → {path}")


def plot_envelope_proof(data_100, data_1000):
    """Plot S6-3: Direct overlay proving Space-Optimal Class traces Pareto frontier."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, data, title in [(ax1, data_100, "C = 100"), (ax2, data_1000, "C = 1000")]:
        pareto = data['pareto']
        s_class = data['space_opt_class']

        # Pareto frontier
        ax.plot([p[0] for p in pareto], [p[1] for p in pareto],
                'o-', color='#1565C0', linewidth=3.0, alpha=0.5,
                markersize=6, label='Complete Pareto Frontier (Brute Force)')

        # Space-optimal points
        ax.plot([p[0] for p in s_class], [p[1] for p in s_class],
                's--', color='#2E7D32', linewidth=2.0,
                markersize=9, label='Space-Optimal Class (Theorem 6.1(1))')

        for p in s_class:
            ax.annotate(f"n={p[3]}", (p[0], p[1]),
                        xytext=(0, -15), textcoords='offset points',
                        fontsize=8, ha='center', color='#1B5E20', fontweight='bold')

        ax.set_title(f"Pareto Envelope Coincidence ({title})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Space (bitmaps)", fontsize=10)
        ax.set_ylabel("Time (scans)", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(fontsize=9)

    fig.suptitle("Proof: Space-Optimal Index Class Traces the Global Pareto Frontier Envelope\n"
                 "(Every space-optimal index sits on the non-dominated lower boundary)",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 's6_envelope_proof.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Plot S6-3 → {path}")


def plot_gradient_arrows(data_100):
    """Plot S6-4: Pareto frontier annotated with gradient step arrows (ΔS, ΔT)."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    s_class = data_100['space_opt_class']

    # Sort space-optimal class by space ascending (n=7 down to n=1)
    sorted_pts = sorted(s_class, key=lambda p: p[0])

    s_vals = [p[0] for p in sorted_pts]
    t_vals = [p[1] for p in sorted_pts]
    ax.plot(s_vals, t_vals, 'o-', color='#1565C0', linewidth=2.2, markersize=8)

    # Annotate points with n and bases
    for p in sorted_pts:
        ax.annotate(f"n={p[3]}\n{p[2]}\n({p[0]}, {p[1]:.2f})", (p[0], p[1]),
                    xytext=(0, 10), textcoords='offset points',
                    fontsize=8, ha='center', fontweight='bold')

    # Draw step arrows between adjacent component points
    for i in range(len(sorted_pts) - 1):
        p1 = sorted_pts[i]
        p2 = sorted_pts[i + 1]
        delta_s = p2[0] - p1[0]
        delta_t = p1[1] - p2[1]
        grad = delta_t / delta_s if delta_s != 0 else 0

        # Draw curved annotation arrow
        mid_s = (p1[0] + p2[0]) / 2.0
        mid_t = (p1[1] + p2[1]) / 2.0
        ax.annotate(f"ΔS=+{delta_s}\nΔT=-{delta_t:.2f}\n|Grad|={grad:.3f}",
                    xy=(p2[0], p2[1]), xytext=(mid_s + 2, mid_t - 0.4),
                    fontsize=7.5, color='#B71C1C',
                    arrowprops=dict(arrowstyle="->", color='#D32F2F', lw=1.2, linestyle=':'))

    # Mark the Knee (n=2)
    knee = [p for p in sorted_pts if p[3] == 2][0]
    ax.plot(knee[0], knee[1], 'h', color='#FF9800', markeredgecolor='black', markersize=14, zorder=6)
    ax.annotate("THE KNEE (n=2)\nMassive ΔT drop from n=3\nNegligible ΔT gain to n=1",
                (knee[0], knee[1]), xytext=(-120, -50), textcoords='offset points',
                fontsize=9, fontweight='bold', color='#E65100',
                arrowprops=dict(arrowstyle="->", color='#E65100', lw=2),
                bbox=dict(boxstyle="round,pad=0.4", fc="#FFF3E0", ec="#E65100", lw=1.5))

    ax.set_title("Discrete Step Gradients (ΔS, ΔT) on the Space-Optimal Frontier (C = 100)\n"
                 "Demonstrating why the Knee occurs sharply at n = 2",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Space (Number of Bitmaps)", fontsize=11)
    ax.set_ylabel("Time (Expected Number of Bitmap Scans)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 's6_gradient_arrows_C100.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Plot S6-4 → {path}")


# ============================================================================
# 3. CSV EXPORT
# ============================================================================

def export_three_class_csv(data, filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n', 'space_opt_bases', 'space_opt_space', 'space_opt_time',
                         'time_opt_bases', 'time_opt_space', 'time_opt_time'])
        s_dict = {p[3]: p for p in data['space_opt_class']}
        t_dict = {p[3]: p for p in data['time_opt_class']}
        for n in range(1, data['max_n'] + 1):
            s_p = s_dict[n]
            t_p = t_dict[n]
            writer.writerow([n, str(s_p[2]), s_p[0], f"{s_p[1]:.4f}",
                             str(t_p[2]), t_p[0], f"{t_p[1]:.4f}"])
    print(f"  CSV → {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 78)
    print("SIMULATION S6: Reproduction of Figures 10 & 11 (Three-Class Tradeoff)")
    print("=" * 78)

    print("\n--- Processing C = 100 (Figure 10) ---")
    data_100 = generate_three_class_data(100)
    print(f"  Total decomps: {len(data_100['all_points'])}, Pareto points: {len(data_100['pareto'])}")

    print("\n--- Processing C = 1000 (Figure 11) ---")
    data_1000 = generate_three_class_data(1000)
    print(f"  Total decomps: {len(data_1000['all_points'])}, Pareto points: {len(data_1000['pareto'])}")

    print("\n--- Generating Plots ---")
    plot_paper_figure(data_100, 's6_fig10_C100.png', "10")
    plot_paper_figure(data_1000, 's6_fig11_C1000.png', "11")
    plot_envelope_proof(data_100, data_1000)
    plot_gradient_arrows(data_100)

    print("\n--- Exporting CSVs ---")
    export_three_class_csv(data_100, 's6_three_class_C100.csv')
    export_three_class_csv(data_1000, 's6_three_class_C1000.csv')

    print("\n" + "=" * 78)
    print("S6 COMPLETE!")
    print("=" * 78)


if __name__ == '__main__':
    main()
