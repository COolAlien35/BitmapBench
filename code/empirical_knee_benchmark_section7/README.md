# Section 7: Empirical Physical Validation of the Knee Point

This subproject implements and reproduces the experiments, mathematical theorems, and visual results for **Section 7: Empirical Physical Validation of the Knee Point** from the foundational paper:
> **"Bitmap Index Design and Evaluation"** — Chee-Yong Chan & Yannis E. Ioannidis (ACM SIGMOD 1998).

---

## 📁 Directory Structure

```text
empirical_knee_benchmark_section7/
├── README.md
├── code/
│   ├── empirical_knee_physical_query_benchmark.py
└── results/
    ├── measured_query_latency_across_frontier_C100.png
    ├── elbow_visualization_speedup_vs_space.png
    ├── latency_breakdown_by_query_predicate.png
    ├── empirical_physical_benchmark_metrics.csv
```

---

## 🚀 Execution

To execute the simulation and regenerate all output plots and datasets:

```bash
python code/empirical_knee_benchmark_section7/code/empirical_knee_physical_query_benchmark.py
```

---

## 📊 Outputs & Artifacts

### Scripts in `code/`:
- `code/empirical_knee_physical_query_benchmark.py`: Main executable script

### Artifacts in `results/`:
- `results/measured_query_latency_across_frontier_C100.png`
- `results/elbow_visualization_speedup_vs_space.png`
- `results/latency_breakdown_by_query_predicate.png`
- `results/empirical_physical_benchmark_metrics.csv`
