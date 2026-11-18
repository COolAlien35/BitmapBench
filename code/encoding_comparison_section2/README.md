# Section 2: Bitmap Index Model, Encodings & Decompositions

This subproject implements and reproduces the experiments, mathematical theorems, and visual results for **Section 2: Bitmap Index Model, Encodings & Decompositions** from the foundational paper:
> **"Bitmap Index Design and Evaluation"** — Chee-Yong Chan & Yannis E. Ioannidis (ACM SIGMOD 1998).

---

## 📁 Directory Structure

```text
encoding_comparison_section2/
├── README.md
├── code/
│   ├── equality_vs_range_encoding_benchmark.py
│   ├── value_list_multicomponent_benchmark.py
│   ├── range_multicomponent_benchmark.py
│   ├── visualize_encoding_tradeoffs.py
└── results/
    ├── bitmap_count_comparison.png
    ├── storage_footprint_comparison.png
    ├── query_latency_by_predicate.png
    ├── cache_sensitivity_hot_vs_cold.png
    ├── logical_operations_breakdown.png
    ├── space_time_tradeoff_curve.png
    ├── equality_vs_range_tradeoff.png
    ├── single_vs_multicomponent_equality.png
    ├── single_vs_multicomponent_range.png
    ├── build_times_summary.csv
    ├── operation_counts_by_predicate.csv
    ├── space_consumption_summary.csv
```

---

## 🚀 Execution

To execute the simulation and regenerate all output plots and datasets:

```bash
python code/encoding_comparison_section2/code/equality_vs_range_encoding_benchmark.py
```

---

## 📊 Outputs & Artifacts

### Scripts in `code/`:
- `code/equality_vs_range_encoding_benchmark.py`: Main executable script
- `code/value_list_multicomponent_benchmark.py`: Main executable script
- `code/range_multicomponent_benchmark.py`: Main executable script
- `code/visualize_encoding_tradeoffs.py`: Main executable script

### Artifacts in `results/`:
- `results/bitmap_count_comparison.png`
- `results/storage_footprint_comparison.png`
- `results/query_latency_by_predicate.png`
- `results/cache_sensitivity_hot_vs_cold.png`
- `results/logical_operations_breakdown.png`
- `results/space_time_tradeoff_curve.png`
- `results/equality_vs_range_tradeoff.png`
- `results/single_vs_multicomponent_equality.png`
- `results/single_vs_multicomponent_range.png`
- `results/build_times_summary.csv`
- `results/operation_counts_by_predicate.csv`
- `results/space_consumption_summary.csv`
