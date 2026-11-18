# Section 3.1: Range Evaluation Algorithms (RangeEval vs RangeEval-Opt)

This subproject implements and reproduces the experiments, mathematical theorems, and visual results for **Section 3.1: Range Evaluation Algorithms (RangeEval vs RangeEval-Opt)** from the foundational paper:
> **"Bitmap Index Design and Evaluation"** — Chee-Yong Chan & Yannis E. Ioannidis (ACM SIGMOD 1998).

---

## 📁 Directory Structure

```text
range_eval_algorithms_section3_1/
├── README.md
├── code/
│   ├── rangeeval_vs_rangeeval_opt_benchmark.py
│   ├── cache_aware_range_evaluation_study.py
└── results/
    ├── table1_worst_case_operations_by_query_type.csv
    ├── table1_worst_case_operations_chart.png
    ├── figure7_rangeeval_opt_execution_trace.png
    ├── rangeeval_vs_rangeeval_opt_ops_comparison.png
    ├── sweep_operations_by_component_count.csv
    ├── cache_aware_query_time_comparison.png
    ├── cache_sensitivity_hot_vs_cold.png
    ├── cardinality_impact_comparison.png
    ├── comprehensive_benchmark_metrics.csv
    ├── bitmap_scans_vs_base_number.png
    ├── bitmap_operations_vs_base_number.png
```

---

## 🚀 Execution

To execute the simulation and regenerate all output plots and datasets:

```bash
python code/range_eval_algorithms_section3_1/code/rangeeval_vs_rangeeval_opt_benchmark.py
```

---

## 📊 Outputs & Artifacts

### Scripts in `code/`:
- `code/rangeeval_vs_rangeeval_opt_benchmark.py`: Main executable script
- `code/cache_aware_range_evaluation_study.py`: Main executable script

### Artifacts in `results/`:
- `results/table1_worst_case_operations_by_query_type.csv`
- `results/table1_worst_case_operations_chart.png`
- `results/figure7_rangeeval_opt_execution_trace.png`
- `results/rangeeval_vs_rangeeval_opt_ops_comparison.png`
- `results/sweep_operations_by_component_count.csv`
- `results/cache_aware_query_time_comparison.png`
- `results/cache_sensitivity_hot_vs_cold.png`
- `results/cardinality_impact_comparison.png`
- `results/comprehensive_benchmark_metrics.csv`
- `results/bitmap_scans_vs_base_number.png`
- `results/bitmap_operations_vs_base_number.png`
