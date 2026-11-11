
import random
import time
import statistics
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to prevent GUI blocking
import matplotlib.pyplot as plt
import numpy as np

# ==================================================
# 1. CONFIGURATION
# ==================================================

N = 5_000_000
C = 18

# Base decomposition:
# Paper notation: Base-<6,3>
# Low-order component: base 3
# High-order component: base 6
BASES = [3, 6]

random.seed(42)

print("=" * 70)
print("NORMAL vs BASE-<6,3> VALUE-LIST INDEX — CACHE-AWARE EVALUATION")
print("=" * 70)

print("Records:", f"{N:,}")
print("Distinct values:", C)
print("Decomposed bases:", BASES)


# ==================================================
# 2. GENERATE DATABASE
# ==================================================

print("\nGenerating database...")
data = bytearray(random.randrange(C) for _ in range(N))


# ==================================================
# 3. HELPER: BUILD BITMAPS EFFICIENTLY
# ==================================================

def create_empty_bitmaps(count, N):
    byte_count = (N + 7) // 8
    return [bytearray(byte_count) for _ in range(count)]


def set_bit(bitmap, record_id):
    bitmap[record_id // 8] |= 1 << (record_id % 8)


def convert_to_integers(buffers):
    return [
        int.from_bytes(b, byteorder="little")
        for b in buffers
    ]


# ==================================================
# 4. NORMAL VALUE-LIST INDEX
# ==================================================

def build_normal_index(data, C):
    buffers = create_empty_bitmaps(C, len(data))
    for record_id, value in enumerate(data):
        set_bit(buffers[value], record_id)
    return convert_to_integers(buffers)


# ==================================================
# 5. BASE-<6,3> VALUE-LIST INDEX
# ==================================================

def build_decomposed_index(data):
    # Three bitmaps for base-3 component
    component1 = create_empty_bitmaps(3, len(data))
    # Six bitmaps for base-6 component
    component2 = create_empty_bitmaps(6, len(data))

    for record_id, value in enumerate(data):
        # v = 3 * digit2 + digit1
        digit1 = value % 3
        digit2 = value // 3

        set_bit(component1[digit1], record_id)
        set_bit(component2[digit2], record_id)

    return (
        convert_to_integers(component1),
        convert_to_integers(component2)
    )


# ==================================================
# 6. BUILD BOTH INDEXES
# ==================================================

print("\nBuilding Normal Value-List Index...")
start = time.perf_counter()
normal_index = build_normal_index(data, C)
normal_build_time = time.perf_counter() - start

print("Building Base-<6,3> Index...")
start = time.perf_counter()
decomposed_index = build_decomposed_index(data)
decomposed_build_time = time.perf_counter() - start


# ==================================================
# 7. QUERY FUNCTIONS
# ==================================================

def normal_query(index, value):
    # One bitmap retrieval
    return index[value]


def decomposed_query(index, value):
    component1, component2 = index
    digit1 = value % 3
    digit2 = value // 3
    # Combine component bitmaps via bitwise AND
    return component1[digit1] & component2[digit2]


# ==================================================
# 8. VERIFY INDEX CORRECTNESS
# ==================================================

print("\nVerifying indexes...")
for value in range(C):
    result1 = normal_query(normal_index, value)
    result2 = decomposed_query(decomposed_index, value)
    assert result1 == result2

print("All 18 equality queries verified successfully!")


# ==================================================
# 9. CACHE-AWARE QUERY BENCHMARKING
# ==================================================
# Caching bias occurs when queries are repeatedly executed without clearing CPU caches.
# We evaluate BOTH:
#   1. Cold Cache: Evicts L1, L2, and L3/SLC cache lines before each measurement.
#   2. Hot Cache:  Warms up CPU cache and executes in steady-state memory.

EVICTION_BUFFER_SIZE = 32 * 1024 * 1024  # 32 MiB
_eviction_buffer = bytearray(EVICTION_BUFFER_SIZE)

def evict_cpu_cache():
    _eviction_buffer[0] = (_eviction_buffer[0] + 1) & 0xFF
    return sum(_eviction_buffer[::64])


def benchmark_cold(query_function, trials=15):
    times = []
    for _ in range(trials):
        evict_cpu_cache()
        start = time.perf_counter()
        _ = query_function()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return statistics.median(times) * 1000  # Return milliseconds per query


def benchmark_hot(query_function, warmup=15, iterations=100, rounds=5):
    for _ in range(warmup):
        query_function()
    times = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(iterations):
            _ = query_function()
        elapsed = time.perf_counter() - start
        times.append(elapsed / iterations)
    return statistics.median(times) * 1000  # Return milliseconds per query


print("\nRunning cache-aware query benchmarks across all 18 values...")

normal_cold_times = []
normal_hot_times = []
decomposed_cold_times = []
decomposed_hot_times = []

print(f"{'Value':<6} | {'Normal Cold (ms)':<17} | {'Normal Hot (ms)':<16} | {'Base-<6,3> Cold (ms)':<21} | {'Base-<6,3> Hot (ms)':<20}")
print("-" * 88)

for value in range(C):
    n_fn = lambda v=value: normal_query(normal_index, v)
    nc = benchmark_cold(n_fn)
    nh = benchmark_hot(n_fn)

    d_fn = lambda v=value: decomposed_query(decomposed_index, v)
    dc = benchmark_cold(d_fn)
    dh = benchmark_hot(d_fn)

    normal_cold_times.append(nc)
    normal_hot_times.append(nh)
    decomposed_cold_times.append(dc)
    decomposed_hot_times.append(dh)

    print(f"A = {value:<2} | {nc:<17.5f} | {nh:<16.5f} | {dc:<21.5f} | {dh:<20.5f}")


# ==================================================
# 10. STORAGE CALCULATION
# ==================================================

normal_bitmaps = C
decomposed_bitmaps = sum(BASES)

normal_storage = normal_bitmaps * N
decomposed_storage = decomposed_bitmaps * N

normal_mb = normal_storage / (8 * 1024 * 1024)
decomposed_mb = decomposed_storage / (8 * 1024 * 1024)

reduction = ((normal_storage - decomposed_storage) / normal_storage) * 100


# ==================================================
# 11. AGGREGATE RESULTS
# ==================================================

avg_norm_cold = statistics.mean(normal_cold_times)
avg_norm_hot  = statistics.mean(normal_hot_times)
avg_dec_cold  = statistics.mean(decomposed_cold_times)
avg_dec_hot   = statistics.mean(decomposed_hot_times)

results_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(results_dir, exist_ok=True)


# ==================================================
# 12. VISUALIZATION 1: COMPREHENSIVE COMPARISON
# ==================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle(
    "Normal (Base-18) vs Base-<6,3> Value-List Index\nCold vs. Hot Cache Performance & Space Analysis",
    fontsize=16,
    fontweight="bold"
)

# ---- PANEL 1: Average Query Latency (Cold vs Hot) ----
labels = ["Normal (Base-18)", "Base-<6,3>"]
x = np.arange(len(labels))
width = 0.35

cold_vals = [avg_norm_cold, avg_dec_cold]
hot_vals  = [avg_norm_hot, avg_dec_hot]

rects1 = axes[0, 0].bar(x - width/2, cold_vals, width, label="Cold Cache (Evicted)", color="#4C72B0", alpha=0.9)
rects2 = axes[0, 0].bar(x + width/2, hot_vals, width, label="Hot Cache (Primed)", color="#55A868", alpha=0.9)

axes[0, 0].set_title("Average Single Query Latency (Cold vs Hot)")
axes[0, 0].set_ylabel("Latency (milliseconds)")
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(labels)
axes[0, 0].legend()
axes[0, 0].grid(axis="y", linestyle="--", alpha=0.4)

for rect in list(rects1) + list(rects2):
    h = rect.get_height()
    axes[0, 0].text(rect.get_x() + rect.get_width()/2., h * 1.02, f"{h:.5f} ms",
                    ha='center', va='bottom', fontsize=8.5)

# ---- PANEL 2: Query Latency Across Values ----
vals = range(C)
axes[0, 1].plot(vals, normal_cold_times, "o--", color="#4C72B0", label="Normal (Cold)", alpha=0.7)
axes[0, 1].plot(vals, normal_hot_times,  "o-",  color="#2b4c7e", label="Normal (Hot)")
axes[0, 1].plot(vals, decomposed_cold_times, "s--", color="#C44E52", label="Base-<6,3> (Cold)", alpha=0.7)
axes[0, 1].plot(vals, decomposed_hot_times,  "s-",  color="#8C2B2F", label="Base-<6,3> (Hot)")

axes[0, 1].set_title("Query Latency Across Values (0 to 17)")
axes[0, 1].set_xlabel("Query value (A = v)")
axes[0, 1].set_ylabel("Latency (milliseconds)")
axes[0, 1].legend(fontsize=8.5)
axes[0, 1].grid(linestyle="--", alpha=0.4)

# ---- PANEL 3: Index Construction Time ----
axes[1, 0].bar(labels, [normal_build_time, decomposed_build_time], color=["#4C72B0", "#DD8452"], alpha=0.85, width=0.5)
axes[1, 0].set_title("Index Construction Time")
axes[1, 0].set_ylabel("Time (seconds)")
axes[1, 0].grid(axis="y", linestyle="--", alpha=0.4)
for i, v in enumerate([normal_build_time, decomposed_build_time]):
    axes[1, 0].text(i, v * 1.01, f"{v:.3f} s", ha='center', va='bottom', fontsize=10)

# ---- PANEL 4: Storage Comparison ----
axes[1, 1].bar(labels, [normal_mb, decomposed_mb], color=["#4C72B0", "#DD8452"], alpha=0.85, width=0.5)
axes[1, 1].set_title(f"Storage Footprint ({reduction:.1f}% space reduction)")
axes[1, 1].set_ylabel("Storage (MiB)")
axes[1, 1].grid(axis="y", linestyle="--", alpha=0.4)
for i, v in enumerate([normal_mb, decomposed_mb]):
    axes[1, 1].text(i, v * 1.01, f"{v:.2f} MiB\n({normal_bitmaps if i==0 else decomposed_bitmaps} bitmaps)",
                    ha='center', va='bottom', fontsize=9.5)

plt.tight_layout()
out_plot1 = os.path.join(results_dir, "normal_vs_base63_comparison.png")
plt.savefig(out_plot1, dpi=300)
print(f"\nPlot saved → {out_plot1}")
plt.close(fig)


# ==================================================
# 13. VISUALIZATION 2: SPACE-TIME TRADEOFF
# ==================================================

fig2, ax = plt.subplots(figsize=(10, 6.5))

# Plot Cold Tradeoff Line
ax.plot(
    [normal_mb, decomposed_mb],
    [avg_norm_cold, avg_dec_cold],
    marker="o",
    linewidth=2.5,
    linestyle="--",
    color="#C44E52",
    label="Cold Cache Frontier (Memory-Bound)"
)

# Plot Hot Tradeoff Line
ax.plot(
    [normal_mb, decomposed_mb],
    [avg_norm_hot, avg_dec_hot],
    marker="s",
    linewidth=2.5,
    linestyle="-",
    color="#4C72B0",
    label="Hot Cache Frontier (CPU/In-Memory)"
)

# Annotations
ax.annotate(
    f"Normal Base-18 (Cold)\n{normal_mb:.1f} MiB, {avg_norm_cold:.4f} ms",
    (normal_mb, avg_norm_cold),
    xytext=(10, -10),
    textcoords="offset points",
    fontsize=9
)
ax.annotate(
    f"Normal Base-18 (Hot)\n{normal_mb:.1f} MiB, {avg_norm_hot:.4f} ms",
    (normal_mb, avg_norm_hot),
    xytext=(10, 10),
    textcoords="offset points",
    fontsize=9
)
ax.annotate(
    f"Base-<6,3> (Cold)\n{decomposed_mb:.1f} MiB, {avg_dec_cold:.4f} ms",
    (decomposed_mb, avg_dec_cold),
    xytext=(-140, 12),
    textcoords="offset points",
    fontsize=9
)
ax.annotate(
    f"Base-<6,3> (Hot)\n{decomposed_mb:.1f} MiB, {avg_dec_hot:.4f} ms",
    (decomposed_mb, avg_dec_hot),
    xytext=(-140, -18),
    textcoords="offset points",
    fontsize=9
)

ax.set_xlabel("Storage Footprint (MiB)", fontsize=11)
ax.set_ylabel("Average Query Latency (ms)", fontsize=11)
ax.set_title("Bitmap Index Space-Time Tradeoff Curve\nCold Cache vs. Hot Cache Regimes", fontsize=14, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(fontsize=10.5, loc="center right")

plt.tight_layout()
out_plot2 = os.path.join(results_dir, "space_time_tradeoff.png")
plt.savefig(out_plot2, dpi=300)
print(f"Plot saved → {out_plot2}")
plt.close(fig2)


# ==================================================
# 14. FINAL REPORT
# ==================================================

print("\n" + "=" * 70)
print("FINAL BENCHMARK REPORT — COLD vs HOT CACHING")
print("=" * 70)

print("\n1. NORMAL VALUE-LIST (Base-18):")
print(f"   • Bitmaps         : {normal_bitmaps}")
print(f"   • Storage         : {normal_mb:.3f} MiB ({normal_storage:,} bits)")
print(f"   • Build Time      : {normal_build_time:.3f} seconds")
print(f"   • Cold Query Time : {avg_norm_cold:.5f} ms ({avg_norm_cold * 1000:.2f} µs)")
print(f"   • Hot Query Time  : {avg_norm_hot:.5f} ms ({avg_norm_hot * 1000:.2f} µs)")
print(f"   • Cache Speedup   : {avg_norm_cold / avg_norm_hot:.2f}x faster when cached")

print("\n2. BASE-<6,3> DECOMPOSED VALUE-LIST:")
print(f"   • Bitmaps         : {decomposed_bitmaps}")
print(f"   • Storage         : {decomposed_mb:.3f} MiB ({decomposed_storage:,} bits)")
print(f"   • Build Time      : {decomposed_build_time:.3f} seconds")
print(f"   • Cold Query Time : {avg_dec_cold:.5f} ms ({avg_dec_cold * 1000:.2f} µs)")
print(f"   • Hot Query Time  : {avg_dec_hot:.5f} ms ({avg_dec_hot * 1000:.2f} µs)")
print(f"   • Cache Speedup   : {avg_dec_cold / avg_dec_hot:.2f}x faster when cached")

print("\n3. ARCHITECTURAL TRADEOFF ANALYSIS:")
print(f"   • Storage Savings : {reduction:.2f}% reduction (from {normal_mb:.2f} MiB down to {decomposed_mb:.2f} MiB)")
print(f"   • Cold Latency Ratio: Base-<6,3> is {avg_dec_cold / avg_norm_cold:.1f}x slower under cold DRAM retrieval")
print(f"   • Hot Latency Ratio : Base-<6,3> is {avg_dec_hot / avg_norm_hot:.1f}x slower under hot CPU cache")
print("=" * 70)
print("EXPERIMENT COMPLETED SUCCESSFULLY!")
