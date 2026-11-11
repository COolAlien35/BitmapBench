"""
plot_comparison_results.py
--------------------------
Runs the Normal vs Base-3 Value-List Index benchmark with CACHE-AWARE
evaluations (Cold Cache vs Hot Cache) and visualizes:
  1. Query Latency per Value (Cold vs Hot Cache for both index types)
  2. Latency Trends across Values (0 to 8)
  3. Construction / Build Time Comparison
  4. Storage Footprint & Theoretical Compression Savings

Requires: matplotlib, numpy
"""

import random
import time
import statistics
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to prevent GUI hangs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ============================================================
# 1. DATABASE CONFIGURATION
# ============================================================

N = 500_000
C = 9

sample_data = [3, 2, 1, 2, 8, 2, 2, 0, 7, 5, 6, 4]

random.seed(42)
data = [random.randrange(C) for _ in range(N)]
data[:12] = sample_data

print("=" * 70)
print("GENERATING COLD & HOT CACHE VISUALIZATIONS (NORMAL vs BASE-3)")
print("=" * 70)


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
# 4. BUILD BOTH INDEXES (timed)
# ============================================================

print("Building indexes...")

t0 = time.perf_counter()
normal_index = build_normal_index(data, C)
normal_build_time = time.perf_counter() - t0

t0 = time.perf_counter()
base3_index = build_base3_index(data)
base3_build_time = time.perf_counter() - t0

print(f"  • Normal build : {normal_build_time:.4f} s")
print(f"  • Base-3 build : {base3_build_time:.4f} s")


# ============================================================
# 5. CACHE EVICTION & BENCHMARK HARNESS
# ============================================================

EVICTION_BUFFER_SIZE = 32 * 1024 * 1024  # 32 MiB
_eviction_buffer = bytearray(EVICTION_BUFFER_SIZE)

def evict_cpu_cache():
    _eviction_buffer[0] = (_eviction_buffer[0] + 1) & 0xFF
    return sum(_eviction_buffer[::64])


def benchmark_cold(fn, trials=25):
    times = []
    for _ in range(trials):
        evict_cpu_cache()
        t0 = time.perf_counter()
        _ = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1_000_000  # microseconds


def benchmark_hot(fn, warmup=25, iterations=400, rounds=5):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = fn()
        times.append((time.perf_counter() - t0) / iterations)
    return statistics.median(times) * 1_000_000  # microseconds


print("\nBenchmarking queries across all 9 values (Cold and Hot Cache)...")

normal_cold_times = []
normal_hot_times  = []
base3_cold_times  = []
base3_hot_times   = []

for v in range(C):
    n_fn = lambda v=v: normal_query(normal_index, v)
    b_fn = lambda v=v: base3_query(base3_index, v)

    nc = benchmark_cold(n_fn)
    nh = benchmark_hot(n_fn)
    bc = benchmark_cold(b_fn)
    bh = benchmark_hot(b_fn)

    normal_cold_times.append(nc)
    normal_hot_times.append(nh)
    base3_cold_times.append(bc)
    base3_hot_times.append(bh)

    print(f"  v={v} | Norm Cold={nc:.3f}µs, Hot={nh:.3f}µs | Base-3 Cold={bc:.3f}µs, Hot={bh:.3f}µs")


# ============================================================
# 6. STORAGE NUMBERS
# ============================================================

normal_bitmaps  = C          # 9
base3_bitmaps   = 3 + 3      # 6
normal_storage  = normal_bitmaps * N   # bits
base3_storage   = base3_bitmaps * N    # bits
reduction = ((normal_storage - base3_storage) / normal_storage) * 100


# ============================================================
# 7. GENERATE MULTI-PANEL FIGURE
# ============================================================

values = list(range(C))
x = np.arange(C)
bar_w = 0.20

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle(
    "Normal vs Base-3 Value-List Index\nUnbiased Cold vs. Hot Cache Performance & Space Evaluation",
    fontsize=16,
    fontweight="bold"
)

# ── Panel 1: Query Latency per Value (Grouped 4-Bar Chart) ──
ax = axes[0, 0]
b_nc = ax.bar(x - 1.5 * bar_w, normal_cold_times, bar_w, label="Normal (Cold)", color="#4C72B0", alpha=0.85)
b_nh = ax.bar(x - 0.5 * bar_w, normal_hot_times,  bar_w, label="Normal (Hot)",  color="#5B9BD5", alpha=0.95)
b_bc = ax.bar(x + 0.5 * bar_w, base3_cold_times,  bar_w, label="Base-3 (Cold)", color="#C44E52", alpha=0.85)
b_bh = ax.bar(x + 1.5 * bar_w, base3_hot_times,   bar_w, label="Base-3 (Hot)",  color="#E67E22", alpha=0.95)

ax.set_title("Query Latency per Value (Cold Cache vs Hot Cache)", fontsize=12, fontweight="bold")
ax.set_xlabel("Query Value (A = v)", fontsize=10.5)
ax.set_ylabel("Latency (µs)", fontsize=10.5)
ax.set_xticks(x)
ax.set_xticklabels([f"v={v}" for v in values])
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.4)

# ── Panel 2: Latency Trend across Values (Line Plot) ─────────
ax = axes[0, 1]
ax.plot(values, normal_cold_times, "o--", color="#4C72B0", linewidth=1.8, markersize=5, label="Normal (Cold Cache)")
ax.plot(values, normal_hot_times,  "o-",  color="#1F4E79", linewidth=2.0, markersize=6, label="Normal (Hot Cache)")
ax.plot(values, base3_cold_times,  "s--", color="#C44E52", linewidth=1.8, markersize=5, label="Base-3 (Cold Cache)")
ax.plot(values, base3_hot_times,   "s-",  color="#8C2B2F", linewidth=2.0, markersize=6, label="Base-3 (Hot Cache)")

ax.set_title("Query Latency Trend across Values (0 to 8)", fontsize=12, fontweight="bold")
ax.set_xlabel("Query Value (A = v)", fontsize=10.5)
ax.set_ylabel("Latency (µs)", fontsize=10.5)
ax.set_xticks(values)
ax.legend(fontsize=9.5)
ax.grid(linestyle="--", alpha=0.4)

# ── Panel 3: Build Time (Bar Chart) ─────────────────────────
ax = axes[1, 0]
labels = ["Normal Index", "Base-3 Index"]
bt_vals = [normal_build_time, base3_build_time]
hbars = ax.bar(labels, bt_vals, color=["#4C72B0", "#E67E22"], alpha=0.85, width=0.45)
ax.set_title("Index Construction Time", fontsize=12, fontweight="bold")
ax.set_ylabel("Time (seconds)", fontsize=10.5)
ax.grid(axis="y", linestyle="--", alpha=0.4)
for bar, val in zip(hbars, bt_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val * 1.02,
            f"{val:.4f} s", ha="center", va="bottom", fontsize=10, fontweight="bold")

# ── Panel 4: Storage Comparison (Grouped Bar Chart) ─────────
ax = axes[1, 1]
storage_labels = ["Total Bitmaps", "Storage (Mbits)"]
normal_vals    = [normal_bitmaps, normal_storage / 1e6]
base3_vals     = [base3_bitmaps,  base3_storage  / 1e6]
xi = np.arange(len(storage_labels))
b1 = ax.bar(xi - 0.18, normal_vals, 0.35, label="Normal Index", color="#4C72B0", alpha=0.85)
b2 = ax.bar(xi + 0.18, base3_vals,  0.35, label="Base-3 Index", color="#E67E22", alpha=0.85)

ax.set_title(f"Storage Comparison ({reduction:.1f}% space reduction with Base-3)", fontsize=12, fontweight="bold")
ax.set_xticks(xi)
ax.set_xticklabels(storage_labels, fontsize=10.5)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

# ── Save Output ──────────────────────────────────────────────
plt.tight_layout()
results_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, "index_comparison.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"\nPlot saved → {out_path}")
plt.close(fig)

print("Visualization completed successfully!")
