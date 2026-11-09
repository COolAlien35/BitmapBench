"""
plot_comparison_results.py
--------------------------
Runs the Normal vs Base-3 Value-List Index benchmark and plots:
  1. Query time per value  (bar chart)
  2. Build time comparison (bar chart)
  3. Storage comparison    (bar chart)
  4. Query time trend      (line chart)

Requires: matplotlib  →  pip install matplotlib
"""

import random
import time
import statistics
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ============================================================
# 1. DATABASE CONFIGURATION  (same as time.py)
# ============================================================

N = 500_000
C = 9

sample_data = [3, 2, 1, 2, 8, 2, 2, 0, 7, 5, 6, 4]

random.seed(42)
data = [random.randrange(C) for _ in range(N)]
data[:12] = sample_data


# ============================================================
# 2. INDEX BUILDERS
# ============================================================

def build_normal_index(data, C):
    bitmaps = [0] * C
    for record_id, value in enumerate(data):
        bitmaps[value] |= (1 << record_id)
    return bitmaps


def build_base3_index(data):
    component1 = [0, 0, 0]
    component2 = [0, 0, 0]
    for record_id, value in enumerate(data):
        digit1 = value % 3
        digit2 = value // 3
        component1[digit1] |= (1 << record_id)
        component2[digit2] |= (1 << record_id)
    return component1, component2


# ============================================================
# 3. QUERY FUNCTIONS
# ============================================================

def normal_query(index, value):
    return index[value]


def base3_query(index, value):
    component1, component2 = index
    return component1[value % 3] & component2[value // 3]


# ============================================================
# 4. BUILD BOTH INDEXES  (timed)
# ============================================================

print("Building indexes…")

t0 = time.perf_counter()
normal_index = build_normal_index(data, C)
normal_build_time = time.perf_counter() - t0

t0 = time.perf_counter()
base3_index = build_base3_index(data)
base3_build_time = time.perf_counter() - t0

print(f"  Normal build : {normal_build_time:.4f}s")
print(f"  Base-3 build : {base3_build_time:.4f}s")


# ============================================================
# 5. BENCHMARK QUERY TIME
# ============================================================

def benchmark(fn, iterations=500, rounds=7):
    for _ in range(20):          # warm-up
        fn()
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iterations):
            fn()
        times.append((time.perf_counter() - t0) / iterations)
    return statistics.median(times) * 1_000_000   # → microseconds


print("Benchmarking queries…")

normal_times = []
base3_times  = []

for v in range(C):
    t1 = benchmark(lambda v=v: normal_query(normal_index, v))
    t2 = benchmark(lambda v=v: base3_query(base3_index,  v))
    normal_times.append(t1)
    base3_times.append(t2)
    print(f"  v={v}  normal={t1:.3f} µs  base3={t2:.3f} µs")


# ============================================================
# 6. STORAGE NUMBERS
# ============================================================

normal_bitmaps  = C          # 9
base3_bitmaps   = 3 + 3      # 6
normal_storage  = normal_bitmaps * N   # bits
base3_storage   = base3_bitmaps * N    # bits


# ============================================================
# 7. PLOT
# ============================================================

values   = list(range(C))
x        = np.arange(C)
bar_w    = 0.35

NORMAL_COLOR = "#4C72B0"
BASE3_COLOR  = "#DD8452"

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Normal vs Base-3 Value-List Index\nPerformance & Space Comparison",
             fontsize=15, fontweight="bold", y=1.01)

# ── Panel 1: Query time per value (grouped bar) ─────────────
ax = axes[0, 0]
bars1 = ax.bar(x - bar_w / 2, normal_times, bar_w,
               label="Normal", color=NORMAL_COLOR, alpha=0.85)
bars2 = ax.bar(x + bar_w / 2, base3_times,  bar_w,
               label="Base-3", color=BASE3_COLOR,  alpha=0.85)
ax.set_title("Query Time per Value")
ax.set_xlabel("Query value  (A = v)")
ax.set_ylabel("Median time  (µs)")
ax.set_xticks(x)
ax.set_xticklabels([f"v={v}" for v in values])
ax.legend()
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.grid(axis="y", linestyle="--", alpha=0.4)
# annotate bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0003,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6.5)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0003,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6.5)

# ── Panel 2: Query time trend (line chart) ──────────────────
ax = axes[0, 1]
ax.plot(values, normal_times, "o-", color=NORMAL_COLOR, linewidth=2,
        markersize=6, label="Normal")
ax.plot(values, base3_times,  "s--", color=BASE3_COLOR, linewidth=2,
        markersize=6, label="Base-3")
ax.fill_between(values, normal_times, base3_times,
                alpha=0.12, color="gray")
ax.set_title("Query Time Trend across Values")
ax.set_xlabel("Query value  (A = v)")
ax.set_ylabel("Median time  (µs)")
ax.set_xticks(values)
ax.legend()
ax.grid(linestyle="--", alpha=0.4)

# ── Panel 3: Build time (horizontal bar) ────────────────────
ax = axes[1, 0]
labels    = ["Normal", "Base-3"]
bt_values = [normal_build_time, base3_build_time]
colors    = [NORMAL_COLOR, BASE3_COLOR]
hbars = ax.barh(labels, bt_values, color=colors, alpha=0.85)
ax.set_title("Index Build Time")
ax.set_xlabel("Time  (seconds)")
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
ax.grid(axis="x", linestyle="--", alpha=0.4)
for bar, val in zip(hbars, bt_values):
    ax.text(val + max(bt_values) * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f} s", va="center", fontsize=10)

# ── Panel 4: Storage (grouped bar) ──────────────────────────
ax = axes[1, 1]
storage_labels = ["# Bitmaps", "Storage (Mbits)"]
normal_vals    = [normal_bitmaps, normal_storage / 1e6]
base3_vals     = [base3_bitmaps,  base3_storage  / 1e6]
xi = np.arange(len(storage_labels))
b1 = ax.bar(xi - bar_w / 2, normal_vals, bar_w,
            label="Normal", color=NORMAL_COLOR, alpha=0.85)
b2 = ax.bar(xi + bar_w / 2, base3_vals,  bar_w,
            label="Base-3", color=BASE3_COLOR,  alpha=0.85)
ax.set_title("Storage Comparison")
ax.set_xticks(xi)
ax.set_xticklabels(storage_labels)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
reduction = (normal_storage - base3_storage) / normal_storage * 100
ax.set_title(f"Storage Comparison  ({reduction:.1f}% reduction with Base-3)")
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

# ── Final layout ─────────────────────────────────────────────
plt.tight_layout()
results_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, "index_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {out_path}")
