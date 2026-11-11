# BitmapBench — Bitmap Index Design and Evaluation

A benchmark and analysis framework evaluating bitmap indexing techniques for database management systems, based on the foundational research paper:
> **"Bitmap Index Design and Evaluation"** — Chee-Yong Chan and Yannis E. Ioannidis (SIGMOD 1998)

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
│   ├── equality_range_comparison.py     # Main simulation: Equality vs Range Encoding (all 6 query types, Hot/Cold)
│   ├── equality_encoding_comparison.py  # Equality index decomposition (Base-9 vs Base-3)
│   ├── range_encoding_comparison.py     # Large-scale decomposed evaluation (Base-18 vs Base-<6,3>)
│   └── plot_comparison_results.py       # Visualizer for equality decomposition
│
├── results/
│   ├── plot1_bitmap_count.png           # Bitmaps: Equality vs Range (Base-18 & Base-<6,3>)
│   ├── plot2_storage_comparison.png     # Theoretical storage comparison
│   ├── plot3_query_time_all_types.png   # Query latency across all 6 query types (Hot Cache)
│   ├── plot4_hot_vs_cold_cache.png      # Side-by-side Hot vs Cold cache latency
│   ├── plot5_bitmap_operations.png      # Average logical bitmap operations (AND/OR/NOT/Scans)
│   ├── plot6_space_time_tradeoff.png    # Storage vs query latency Pareto frontier
│   ├── benchmark_results.csv            # Raw timing benchmarks (Hot & Cold)
│   ├── operation_counts.csv             # Bitmap operation and scan metrics
│   ├── space_comparison.csv             # Storage footprint in bits, bytes, KiB, MiB
│   ├── build_times.csv                  # Index construction durations
│   ├── index_comparison.png             # Legacy equality plot
│   ├── normal_vs_base63_comparison.png  # Legacy 5M record plot
│   └── space_time_tradeoff.png          # Legacy tradeoff plot
│
└── .gitignore
```

---

## 🧠 Encoding Schemes & The Space-Time Tradeoff

From Section 2 of Chan & Ioannidis (SIGMOD 1998):

| Feature | Equality Encoding | Range Encoding |
| :--- | :--- | :--- |
| **Bitmap Definition** | $B_i[r] = 1 \iff \text{val}(r) = i$ | $B_i[r] = 1 \iff \text{val}(r) \le i$ |
| **Bitmaps per component ($b$)** | $b$ bitmaps | $b - 1$ bitmaps ($B_{b-1}$ is all-ones, omitted) |
| **Point Queries ($A = v$)** | Direct single bitmap lookup ($0$ ops) | $B_v \land \neg B_{v-1}$ ($1$ AND, $1$ NOT) |
| **Range Queries ($A \le v$, $A < v$)** | Cumulative OR of $v$ bitmaps ($O(v)$ ops) | Direct single bitmap lookup ($0$ ops) |
| **Multi-Component Base-$\langle b_2, b_1 \rangle$** | $b_1 + b_2$ bitmaps | $(b_1 - 1) + (b_2 - 1)$ bitmaps |

---

## 🔬 Simulations Overview

| File | Description | Configurations Tested |
| :--- | :--- | :--- |
| [`code/equality_range_comparison.py`](code/equality_range_comparison.py) | **Full Equality vs. Range Comparison** | 4 configs: Equality Base-18, Equality Base-$\langle 6,3 \rangle$, Range Base-18, Range Base-$\langle 6,3 \rangle$ on $N = 100,000$ records, testing all 6 query types ($<, \le, >, \ge, =, \ne$) under both Hot and Cold cache conditions. |
| [`code/equality_encoding_comparison.py`](code/equality_encoding_comparison.py) | Equality decomposition | Normal Base-9 vs. Base-3 ($N = 500,000, C = 9$) |
| [`code/range_encoding_comparison.py`](code/range_encoding_comparison.py) | Large-scale decomposed base | Normal Base-18 vs. Base-$\langle 6,3 \rangle$ ($N = 5,000,000, C = 18$) |
| [`code/plot_comparison_results.py`](code/plot_comparison_results.py) | Visualizer | Generates multi-panel summary of Base-3 experiment |

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install matplotlib numpy

# 2. Run the main Equality vs Range experiment (generates 6 plots and 4 CSVs)
python3 code/equality_range_comparison.py

# 3. Output files will be stored in results/
ls -l results/
```

---

## 📊 Experimental Results ($N = 100,000$, $C = 18$)

### Space Footprint
| Index Configuration | Bitmaps | Storage (KiB) | Storage Reduction |
| :--- | :---: | :---: | :---: |
| **Equality Base-18** | 18 | 219.7 KiB | Baseline ($0.0\%$) |
| **Range Base-18** | 17 | 207.5 KiB | **$5.6\%$ saved** |
| **Equality Base-$\langle 6,3 \rangle$** | 9 | 109.9 KiB | **$50.0\%$ saved** |
| **Range Base-$\langle 6,3 \rangle$** | **7** | **85.4 KiB** | **$61.1\%$ saved (Minimal Storage)** |

### Query Latency & Operation Summary
- **Equality Queries ($A = v$)**: Equality Base-18 is fastest ($\sim 0.22\ \mu\text{s}$ hot), requiring only $1$ bitmap read and $0$ Boolean operations.
- **Range Queries ($A < v, A \le v$)**: Range Base-18 is fastest ($\sim 0.26\ \mu\text{s}$ hot), requiring only $1$ bitmap read and $0$ Boolean operations, whereas Equality Base-18 requires averaging $8.5$ OR operations and $9.5$ bitmap scans ($\sim 11.9\ \mu\text{s}$ hot).
- **Cache Sensitivity**: Point lookups show high cache sensitivity ($5\times$ speedup between cold DRAM fetch and hot L1/L2 cache), while decomposed multi-bitmap operations are compute-bound and maintain consistent latency across cache regimes.

---

## 📚 References
- [`paper/bitmap.pdf`](paper/bitmap.pdf): Chan & Ioannidis, *"Bitmap Index Design and Evaluation"*, ACM SIGMOD 1998.
