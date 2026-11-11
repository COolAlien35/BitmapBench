
import random
import time
import statistics

# ============================================
# 1. DATABASE CONFIGURATION
# ============================================

N = 500_000
C = 9

# Same values as Figure 1 and Figure 3
sample_data = [3, 2, 1, 2, 8, 2, 2, 0, 7, 5, 6, 4]

random.seed(42)

# Generate a larger database with values 0 to 8
data = [random.randrange(C) for _ in range(N)]

# Keep the paper's first 12 records
data[:12] = sample_data

print("=" * 70)
print("NORMAL vs BASE-3 VALUE-LIST INDEX — CACHING BENCHMARK")
print("=" * 70)

print("Number of records:", N)
print("Distinct values:", C)


# ============================================
# 2. NORMAL VALUE-LIST INDEX
# ============================================

def build_normal_index(data, C):
    bitmaps = [0] * C
    for record_id, value in enumerate(data):
        bitmaps[value] |= (1 << record_id)
    return bitmaps


# ============================================
# 3. BASE-3 VALUE-LIST INDEX
# ============================================

def build_base3_index(data):
    # Three bitmaps for component 1
    # Three bitmaps for component 2
    component1 = [0, 0, 0]
    component2 = [0, 0, 0]

    for record_id, value in enumerate(data):
        # Decompose value into base-3 digits
        digit1 = value % 3
        digit2 = value // 3

        # Set corresponding bitmap bits
        component1[digit1] |= (1 << record_id)
        component2[digit2] |= (1 << record_id)

    return component1, component2


# ============================================
# 4. BUILD BOTH INDEXES
# ============================================

start = time.perf_counter()
normal_index = build_normal_index(data, C)
normal_build_time = time.perf_counter() - start

start = time.perf_counter()
base3_index = build_base3_index(data)
base3_build_time = time.perf_counter() - start


# ============================================
# 5. DISPLAY NORMAL VALUE-LIST INDEX
# ============================================

print("\nNORMAL VALUE-LIST INDEX (first 12 records)")
print("Record | Value | ", end="")
for v in range(C):
    print(f"B{v} ", end="")
print()

for i in range(12):
    print(f"{i+1:6} | {data[i]:5} | ", end="")
    for v in range(C):
        print(f"{(normal_index[v] >> i) & 1}  ", end="")
    print()


# ============================================
# 6. DISPLAY BASE-3 VALUE-LIST INDEX
# ============================================

print("\nBASE-3 VALUE-LIST INDEX (first 12 records)")
print("Record | Value | Digits | ", end="")
for v in range(3):
    print(f"B2_{v} ", end="")
print("| ", end="")
for v in range(3):
    print(f"B1_{v} ", end="")
print()

for i in range(12):
    value = data[i]
    digit1 = value % 3
    digit2 = value // 3
    print(f"{i+1:6} | {value:5} | ({digit2},{digit1})  | ", end="")
    for v in range(3):
        print(f"{(base3_index[1][v] >> i) & 1}    ", end="")
    print("| ", end="")
    for v in range(3):
        print(f"{(base3_index[0][v] >> i) & 1}    ", end="")
    print()


# ============================================
# 7. QUERY USING NORMAL VALUE-LIST INDEX
# ============================================

def normal_query(index, value):
    # Directly retrieve the bitmap
    return index[value]


# ============================================
# 8. QUERY USING BASE-3 INDEX
# ============================================

def base3_query(index, value):
    component1, component2 = index
    digit1 = value % 3
    digit2 = value // 3
    # Combine the two component bitmaps via bitwise AND
    return component1[digit1] & component2[digit2]


# ============================================
# 9. VERIFY BOTH INDEXES
# ============================================

print("\nVERIFYING QUERY RESULTS")
for value in range(C):
    result1 = normal_query(normal_index, value)
    result2 = base3_query(base3_index, value)
    assert result1 == result2
    print(f"A = {value}: Both indexes return {result1.bit_count()} matching records")

print("All queries produce identical results!")


# ============================================
# 10. CACHE-AWARE BENCHMARK HARNESS
# ============================================
# Modern CPUs feature multi-level caches (L1: ~128KB, L2: ~4-16MB, L3/SLC: ~24-32MB).
# Repeated execution of the same query keeps bitmaps pinned in L1/L2 cache (Hot Cache).
# To prevent cache bias, we measure BOTH:
#   1. Cold Cache: Evict CPU cache before each query to simulate fresh DRAM retrieval.
#   2. Hot Cache:  Warm up CPU cache and measure steady-state in-memory throughput.

EVICTION_BUFFER_SIZE = 32 * 1024 * 1024  # 32 MiB buffer
_eviction_buffer = bytearray(EVICTION_BUFFER_SIZE)

def evict_cpu_cache():
    """Forces CPU L1, L2, and L3/SLC cache eviction by reading and mutating

    a 32 MiB buffer, displacing index bitmaps from CPU cache lines.
    """
    _eviction_buffer[0] = (_eviction_buffer[0] + 1) & 0xFF
    return sum(_eviction_buffer[::64])


def benchmark_cold(query_function, trials=25):
    """Measures query execution under pristine Cold Cache conditions.

    Before every single measurement, the CPU cache hierarchy is flushed.
    """
    times = []
    for _ in range(trials):
        evict_cpu_cache()
        start = time.perf_counter()
        _ = query_function()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return statistics.median(times) * 1_000_000  # Return microseconds


def benchmark_hot(query_function, warmup=25, iterations=400, rounds=5):
    """Measures query execution under primed Hot Cache conditions.

    Cache is primed first, followed by steady-state batched execution.
    """
    # Warm-up: prime L1/L2 cache
    for _ in range(warmup):
        query_function()

    times = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(iterations):
            _ = query_function()
        elapsed = time.perf_counter() - start
        times.append(elapsed / iterations)
    return statistics.median(times) * 1_000_000  # Return microseconds


# ============================================
# 11. EXECUTE COLD & HOT BENCHMARKS
# ============================================

print("\n" + "=" * 70)
print("BENCHMARKING QUERY PERFORMANCE (COLD vs HOT CACHING)")
print("=" * 70)

normal_cold_times = []
normal_hot_times = []
base3_cold_times = []
base3_hot_times = []

print(f"{'Value':<6} | {'Normal Cold (µs)':<17} | {'Normal Hot (µs)':<16} | {'Base-3 Cold (µs)':<17} | {'Base-3 Hot (µs)':<16}")
print("-" * 78)

for value in range(C):
    # Normal Index
    norm_fn = lambda v=value: normal_query(normal_index, v)
    nc = benchmark_cold(norm_fn)
    nh = benchmark_hot(norm_fn)

    # Base-3 Index
    b3_fn = lambda v=value: base3_query(base3_index, v)
    b3c = benchmark_cold(b3_fn)
    b3h = benchmark_hot(b3_fn)

    normal_cold_times.append(nc)
    normal_hot_times.append(nh)
    base3_cold_times.append(b3c)
    base3_hot_times.append(b3h)

    print(f"A = {value:<2} | {nc:<17.3f} | {nh:<16.3f} | {b3c:<17.3f} | {b3h:<16.3f}")


# ============================================
# 12. SUMMARY & CACHE IMPACT ANALYSIS
# ============================================

avg_norm_cold = statistics.mean(normal_cold_times)
avg_norm_hot  = statistics.mean(normal_hot_times)
avg_b3_cold   = statistics.mean(base3_cold_times)
avg_b3_hot    = statistics.mean(base3_hot_times)

print("\n" + "=" * 70)
print("AVERAGE QUERY LATENCY COMPARISON")
print("=" * 70)
print(f"Normal Value-List Index:")
print(f"  • Cold Cache Latency : {avg_norm_cold:.3f} µs")
print(f"  • Hot Cache Latency  : {avg_norm_hot:.3f} µs")
print(f"  • Cache Speedup      : {avg_norm_cold / avg_norm_hot:.2f}x faster when cached")

print(f"\nBase-3 Value-List Index:")
print(f"  • Cold Cache Latency : {avg_b3_cold:.3f} µs")
print(f"  • Hot Cache Latency  : {avg_b3_hot:.3f} µs")
print(f"  • Cache Speedup      : {avg_b3_cold / avg_b3_hot:.2f}x faster when cached")

print("\nRelative Speedup (Normal vs Base-3):")
print(f"  • In Cold Cache: Normal is {avg_b3_cold / avg_norm_cold:.1f}x faster")
print(f"  • In Hot Cache : Normal is {avg_b3_hot / avg_norm_hot:.1f}x faster")


# ============================================
# 13. STORAGE & BUILD TIME COMPARISON
# ============================================

normal_bitmaps = C
base3_bitmaps = 3 + 3

normal_storage = normal_bitmaps * N
base3_storage = base3_bitmaps * N

reduction = ((normal_storage - base3_storage) / normal_storage) * 100

print("\n" + "=" * 70)
print("STORAGE & CONSTRUCTION SUMMARY")
print("=" * 70)
print("Normal Value-List Index:")
print(f"  • Bitmaps : {normal_bitmaps}")
print(f"  • Storage : {normal_storage:,} bits ({normal_storage / (8 * 1024):.2f} KiB)")
print(f"  • Build   : {normal_build_time:.4f} seconds")

print("\nBase-3 Value-List Index:")
print(f"  • Bitmaps : {base3_bitmaps}")
print(f"  • Storage : {base3_storage:,} bits ({base3_storage / (8 * 1024):.2f} KiB)")
print(f"  • Build   : {base3_build_time:.4f} seconds")

print(f"\nStorage Savings: {reduction:.2f}% space reduction with Base-3 decomposition")
print("=" * 70)
