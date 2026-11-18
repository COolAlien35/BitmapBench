# Section 8: Time-Optimal Bitmap Index Under Space Constraint (Table 2 & Figures 12-15)

This subproject implements and reproduces the experiments, mathematical theorems, and visual results for **Section 8: Time-Optimal Bitmap Index Under Space Constraint (Table 2 & Figures 12-15)** from the foundational paper:
> **"Bitmap Index Design and Evaluation"** — Chee-Yong Chan & Yannis E. Ioannidis (ACM SIGMOD 1998).

---

## 📁 Directory Structure

```text
time_optimal_constrained_design_section8/
├── README.md
├── code/
│   ├── time_optimal_index_selection_simulation.py
└── results/
    ├── table2_optimal_vs_heuristic_reproduction.csv
    ├── figure13_heuristic_accuracy_and_scan_difference.png
    ├── figure14_query_time_vs_space_budget_tradeoff.png
    ├── optimal_vs_heuristic_cost_scatter.png
    ├── branch_and_bound_vs_heuristic_scalability.png
    ├── physical_query_latency_vs_theoretical_cost.png
```

---

## 🚀 Execution

To execute the simulation and regenerate all output plots and datasets:

```bash
python code/time_optimal_constrained_design_section8/code/time_optimal_index_selection_simulation.py
```

---

## 📊 Outputs & Artifacts

### Scripts in `code/`:
- `code/time_optimal_index_selection_simulation.py`: Main executable script

### Artifacts in `results/`:
- `results/table2_optimal_vs_heuristic_reproduction.csv`
- `results/figure13_heuristic_accuracy_and_scan_difference.png`
- `results/figure14_query_time_vs_space_budget_tradeoff.png`
- `results/optimal_vs_heuristic_cost_scatter.png`
- `results/branch_and_bound_vs_heuristic_scalability.png`
- `results/physical_query_latency_vs_theoretical_cost.png`
