# BitmapBench — Cache-Aware Bitmap Indexing Evaluation

A benchmark and analysis framework evaluating bitmap indexing techniques for database management systems, comparing standard single-component Value-List indexes against multi-component decomposed base encodings (e.g. Base-3 and Base-$\langle 6,3 \rangle$).

---

## 📁 Directory Structure

```text
BitmapBench/
│
├── README.md
│
├── paper/
│   └── bitmap.pdf
│
├── code/
│   ├── equality_encoding_comparison.py
│   ├── range_encoding_comparison.py
│   └── plot_comparison_results.py
│
├── results/
│   ├── index_comparison.png
│   ├── normal_vs_base63_comparison.png
│   └── space_time_tradeoff.png
│
└── .gitignore
```

---

## 🧠 Why Caching Creates Experimental Bias

In standard in-memory benchmarks, running the same query in a tight loop introduces severe **caching bias**:
1. **L1/L2 Cache Pinning**: Operands remain in high-speed CPU caches (1–10 ns access latency), masking memory bus transfer overheads.
2. **Asymmetric Cache Hits in Decomposed Indexes**: Component bitmaps with smaller cardinalities (e.g., base-3 component in Base-$\langle 6,3 \rangle$) are frequently reused across consecutive queries, artificially inflating their cache hit rate.
3. **Pointers vs. Compute Overhead**: Single-component bitmap retrievals touch fewer cache lines than multi-component bitwise AND operations.

### The Solution: Dual-Regime Benchmarking (Cold vs. Hot Cache)
To ensure completely unbiased results, our benchmarking harness evaluates **both** memory regimes:
- **Cold Cache (DRAM / Memory-Bound)**: Before each query execution, a 32 MiB cache-buster buffer is swept across all 64-byte cache lines, fully displacing cached index bitmaps from L1, L2, and L3/SLC caches. This simulates first-access/ad-hoc query latency directly from DRAM.
- **Hot Cache (In-Memory / CPU-Bound)**: Queries are warmed up and executed in steady-state, measuring peak CPU bitwise arithmetic throughput without DRAM latency.

---

## 🔬 Scripts Overview

| File | Description | What It Compares |
| :--- | :--- | :--- |
| [`code/equality_encoding_comparison.py`](code/equality_encoding_comparison.py) | Equality index comparison | Evaluates standard single-component Value-List index vs. Base-3 decomposed index ($N = 500,000$, $C = 9$) across Cold Cache and Hot Cache regimes. |
| [`code/range_encoding_comparison.py`](code/range_encoding_comparison.py) | Decomposed base index evaluation | Compares Single-component Normal index against two-component decomposed base index (Base-$\langle 6, 3 \rangle$, $N = 5,000,000$, $C = 18$), measuring latency, space-time tradeoffs, and cache sensitivity. |
| [`code/plot_comparison_results.py`](code/plot_comparison_results.py) | Experimental visualization | Generates comprehensive visual comparisons across Cold vs Hot query times, build performance, and storage savings. |

---

## 🚀 How to Run

### 1. Prerequisites
Ensure Python 3.10+ and the required packages are installed:
```bash
pip install matplotlib numpy
```

### 2. Running Benchmarks
Execute any benchmark directly from the project root:

```bash
# Run equality encoding benchmark (Cold vs Hot)
python3 code/equality_encoding_comparison.py

# Run range/decomposed encoding benchmark (Cold vs Hot + Space-Time Tradeoff)
python3 code/range_encoding_comparison.py

# Generate comparison plots
python3 code/plot_comparison_results.py
```

Generated plots will be saved directly into [`results/`](results/).

---

## 📊 Experimental Results

Plots generated from experimental evaluations:

- **Index Comparison**: [`results/index_comparison.png`](results/index_comparison.png)  
  *Query latency per value comparing Normal vs. Base-3 across Cold and Hot cache, construction time, and storage footprint.*
- **Normal vs. Base-$\langle 6,3 \rangle$ Comparison**: [`results/normal_vs_base63_comparison.png`](results/normal_vs_base63_comparison.png)  
  *Comprehensive 4-panel comparison of Normal (Base-18) vs. Base-$\langle 6,3 \rangle$ on 5,000,000 records, highlighting 50% storage reduction and Cold vs. Hot query latencies.*
- **Space-Time Tradeoff**: [`results/space_time_tradeoff.png`](results/space_time_tradeoff.png)  
  *Storage (MiB) vs. Average Query Latency (ms) frontier showing both Cold and Hot Cache operating regimes.*

---

## 📚 References
- [`paper/bitmap.pdf`](paper/bitmap.pdf): Reference research paper on bitmap index encoding and decomposition techniques.
