
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

print("=" * 65)
print("NORMAL vs BASE-3 VALUE-LIST INDEX")
print("=" * 65)

print("Number of records:", N)
print("Distinct values:", C)


# ============================================
# 2. NORMAL VALUE-LIST INDEX
# ============================================

def build_normal_index(data, C):

    N = len(data)
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

print("\nNORMAL VALUE-LIST INDEX")

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

print("\nBASE-3 VALUE-LIST INDEX")

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

    print(
        f"{i+1:6} | {value:5} | "
        f"({digit2},{digit1})  | ",
        end=""
    )

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

    # Combine the two component bitmaps
    result = component1[digit1] & component2[digit2]

    return result


# ============================================
# 9. VERIFY BOTH INDEXES
# ============================================

print("\nVERIFYING QUERY RESULTS")

for value in range(C):

    result1 = normal_query(normal_index, value)

    result2 = base3_query(base3_index, value)

    assert result1 == result2

    print(
        f"A = {value}: "
        f"Both indexes return {result1.bit_count()} records"
    )

print("\nAll queries produce identical results!")


# ============================================
# 10. BENCHMARK QUERY PROCESSING TIME
# ============================================

def benchmark(query_function, iterations=500, rounds=7):

    # Warm-up
    for _ in range(20):
        query_function()

    times = []

    for _ in range(rounds):

        start = time.perf_counter()

        for _ in range(iterations):
            result = query_function()

        elapsed = time.perf_counter() - start

        times.append(elapsed / iterations)

    return statistics.median(times) * 1_000_000


print("\nQUERY PERFORMANCE")

normal_times = []
base3_times = []

for value in range(C):

    t1 = benchmark(
        lambda v=value: normal_query(normal_index, v)
    )

    t2 = benchmark(
        lambda v=value: base3_query(base3_index, v)
    )

    normal_times.append(t1)
    base3_times.append(t2)

    print(f"\nQuery: A = {value}")

    print(f"Normal Value-List: {t1:.3f} microseconds")

    print(f"Base-3 Value-List: {t2:.3f} microseconds")


# ============================================
# 11. AVERAGE QUERY TIME
# ============================================

avg_normal = statistics.mean(normal_times)
avg_base3 = statistics.mean(base3_times)

print("\n" + "=" * 65)
print("AVERAGE QUERY TIME")
print("=" * 65)

print(f"Normal Value-List: {avg_normal:.3f} microseconds")

print(f"Base-3 Value-List: {avg_base3:.3f} microseconds")

if avg_normal < avg_base3:
    print("Normal Value-List is faster for equality lookup.")
else:
    print("Base-3 Value-List is faster for equality lookup.")


# ============================================
# 12. STORAGE COMPARISON
# ============================================

normal_bitmaps = C
base3_bitmaps = 3 + 3

normal_storage = normal_bitmaps * N
base3_storage = base3_bitmaps * N

print("\n" + "=" * 65)
print("STORAGE COMPARISON")
print("=" * 65)

print("Normal Value-List:")
print("Number of bitmaps:", normal_bitmaps)
print("Storage:", normal_storage, "bits")

print("\nBase-3 Value-List:")
print("Number of bitmaps:", base3_bitmaps)
print("Storage:", base3_storage, "bits")

reduction = (
    (normal_storage - base3_storage)
    / normal_storage
) * 100

print(f"\nStorage reduction: {reduction:.2f}%")

print("\nBUILDING TIME")

print(f"Normal index: {normal_build_time:.4f} seconds")

print(f"Base-3 index: {base3_build_time:.4f} seconds")

print("\nSIMULATION COMPLETED!")
