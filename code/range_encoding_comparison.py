
import random
import time
import statistics
import os
import matplotlib.pyplot as plt

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

QUERY_VALUE = 17

random.seed(42)

print("=" * 65)
print("NORMAL vs BASE-<6,3> VALUE-LIST INDEX")
print("=" * 65)

print("Records:", N)
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

    # Combine component bitmaps
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
# 9. BENCHMARK QUERY TIME
# ==================================================

ITERATIONS = 200
ROUNDS = 5


def benchmark(query_function):

    # Warm-up
    for _ in range(3):
        query_function()

    measurements = []

    for _ in range(ROUNDS):

        start = time.perf_counter()

        for _ in range(ITERATIONS):
            result = query_function()

        elapsed = time.perf_counter() - start

        measurements.append(elapsed)

    median_batch_time = statistics.median(measurements)

    # Return milliseconds for ITERATIONS queries
    return median_batch_time * 1000


normal_times = []
decomposed_times = []

print("\nRunning query benchmarks...")

for value in range(C):

    t1 = benchmark(
        lambda v=value: normal_query(normal_index, v)
    )

    t2 = benchmark(
        lambda v=value: decomposed_query(decomposed_index, v)
    )

    normal_times.append(t1)
    decomposed_times.append(t2)

    print(f"\nQuery: A = {value}")

    print(f"Normal: {t1:.4f} ms per {ITERATIONS} queries")

    print(f"Base-<6,3>: {t2:.4f} ms per {ITERATIONS} queries")


# ==================================================
# 10. STORAGE CALCULATION
# ==================================================

normal_bitmaps = C

decomposed_bitmaps = sum(BASES)

normal_storage = normal_bitmaps * N

decomposed_storage = decomposed_bitmaps * N

normal_mb = normal_storage / (8 * 1024 * 1024)

decomposed_mb = decomposed_storage / (8 * 1024 * 1024)

reduction = (
    (normal_storage - decomposed_storage)
    / normal_storage
) * 100


# ==================================================
# 11. PREPARE RESULTS
# ==================================================

avg_normal = statistics.mean(normal_times)

avg_decomposed = statistics.mean(decomposed_times)

normal_per_query = avg_normal / ITERATIONS

decomposed_per_query = avg_decomposed / ITERATIONS


# ==================================================
# 12. CREATE VISUALIZATIONS
# ==================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

fig.suptitle(
    "Normal vs Base-<6,3> Value-List Index",
    fontsize=17,
    fontweight="bold"
)

# ---- GRAPH 1: Query time ----

axes[0, 0].bar(
    ["Normal", "Base-<6,3>"],
    [avg_normal, avg_decomposed]
)

axes[0, 0].set_title(
    f"Average Batch Query Time ({ITERATIONS} queries)"
)

axes[0, 0].set_ylabel("Time (milliseconds)")

axes[0, 0].grid(axis="y", alpha=0.3)


# ---- GRAPH 2: Query time across values ----

axes[0, 1].plot(
    range(C),
    normal_times,
    marker="o",
    label="Normal"
)

axes[0, 1].plot(
    range(C),
    decomposed_times,
    marker="s",
    label="Base-<6,3>"
)

axes[0, 1].set_title("Query Time Across Attribute Values")

axes[0, 1].set_xlabel("Query value")

axes[0, 1].set_ylabel(
    f"Time per {ITERATIONS} queries (ms)"
)

axes[0, 1].legend()

axes[0, 1].grid(alpha=0.3)


# ---- GRAPH 3: Index build time ----

axes[1, 0].bar(
    ["Normal", "Base-<6,3>"],
    [normal_build_time, decomposed_build_time]
)

axes[1, 0].set_title("Index Construction Time")

axes[1, 0].set_ylabel("Time (seconds)")

axes[1, 0].grid(axis="y", alpha=0.3)


# ---- GRAPH 4: Storage comparison ----

axes[1, 1].bar(
    ["Normal", "Base-<6,3>"],
    [normal_mb, decomposed_mb]
)

axes[1, 1].set_title(
    f"Storage Comparison ({reduction:.1f}% reduction)"
)

axes[1, 1].set_ylabel("Storage (MiB)")

axes[1, 1].grid(axis="y", alpha=0.3)


plt.tight_layout()

results_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
os.makedirs(results_dir, exist_ok=True)

out_plot1 = os.path.join(results_dir, "normal_vs_base63_comparison.png")
plt.savefig(out_plot1, dpi=300)
print(f"Plot saved → {out_plot1}")

plt.show()


# ==================================================
# 13. SPACE-TIME TRADEOFF
# ==================================================

plt.figure(figsize=(9, 6))

plt.plot(
    [normal_mb, decomposed_mb],
    [normal_per_query, decomposed_per_query],
    marker="o",
    linewidth=2
)

plt.annotate(
    "Normal Base-18",
    (normal_mb, normal_per_query),
    xytext=(8, 8),
    textcoords="offset points"
)

plt.annotate(
    "Base-<6,3>",
    (decomposed_mb, decomposed_per_query),
    xytext=(8, 8),
    textcoords="offset points"
)

plt.xlabel("Storage (MiB)")

plt.ylabel("Average query time (ms)")

plt.title("Bitmap Index Space-Time Tradeoff")

plt.grid(alpha=0.3)

plt.tight_layout()

out_plot2 = os.path.join(results_dir, "space_time_tradeoff.png")
plt.savefig(out_plot2, dpi=300)
print(f"Plot saved → {out_plot2}")

plt.show()


# ==================================================
# 14. FINAL REPORT
# ==================================================

print("\n" + "=" * 65)
print("FINAL EXPERIMENT RESULTS")
print("=" * 65)

print("\nNORMAL VALUE-LIST")

print("Bitmaps:", normal_bitmaps)

print(f"Storage: {normal_mb:.3f} MiB")

print(f"Build time: {normal_build_time:.3f} seconds")

print(f"Query time: {avg_normal:.4f} ms per batch")


print("\nBASE-<6,3> VALUE-LIST")

print("Bitmaps:", decomposed_bitmaps)

print(f"Storage: {decomposed_mb:.3f} MiB")

print(f"Build time: {decomposed_build_time:.3f} seconds")

print(f"Query time: {avg_decomposed:.4f} ms per batch")


print("\nSTORAGE REDUCTION:", f"{reduction:.2f}%")

print("\nBoth indexes return identical query results.")

print("\nEXPERIMENT COMPLETED!")
