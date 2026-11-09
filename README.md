# Bitmap Indexing Evaluation — DBMS Project

This project implements and benchmarks bitmap indexing techniques for database management systems, based on research into bitmap index compression and component decomposition strategies (e.g., normal value-list vs. multi-component decomposed base encodings).

---

## 📁 Directory Structure

```text
dbms_project/
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

## Scripts Overview

| File | Description | What It Compares |
| :--- | :--- | :--- |
| [`code/equality_encoding_comparison.py`](code/equality_encoding_comparison.py) | Equality index comparison | Evaluates standard single-component Value-List index vs. decomposed component index on equality queries, measuring build times, retrieval times, and storage footprint. |
| [`code/range_encoding_comparison.py`](code/range_encoding_comparison.py) | Decomposed base index evaluation | Compares Single-component Normal index against two-component decomposed base index (e.g., Base-$\langle 6, 3 \rangle$), evaluating query latency and space-time tradeoffs. |
| [`code/plot_comparison_results.py`](code/plot_comparison_results.py) | Experimental visualization | Generates comprehensive visual comparisons across query times, build performance, and storage savings. |

---

## How to Run

### 1. Prerequisites
Ensure Python 3.10+ and the required packages are installed:
```bash
pip install matplotlib numpy
```

### 2. Running Benchmarks
Execute any benchmark directly from the project root:

```bash
# Run equality encoding benchmark
python3 code/equality_encoding_comparison.py

# Run range/decomposed encoding benchmark
python3 code/range_encoding_comparison.py

# Generate comparison plots
python3 code/plot_comparison_results.py
```

Generated plots will be saved into the `results/` folder.

---

## Experimental Results

Plots generated from experimental evaluations:

- **Index Comparison**: [`results/index_comparison.png`](results/index_comparison.png)  
  *Detailed breakdown of query processing latency across values, index building time, and storage bit count.*
- **Normal vs. Base-$\langle 6,3 \rangle$ Comparison**: [`results/normal_vs_base63_comparison.png`](results/normal_vs_base63_comparison.png)  
  *Comparative evaluation of memory footprint and query times across decomposed base encodings.*
- **Space-Time Tradeoff**: [`results/space_time_tradeoff.png`](results/space_time_tradeoff.png)  
  *Storage (MiB) vs. Average Query Time (ms) frontier.*

---

## References
- [`paper/bitmap.pdf`](paper/bitmap.pdf): Reference research paper on bitmap index encoding and decomposition techniques.

