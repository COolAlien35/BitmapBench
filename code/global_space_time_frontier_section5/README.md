# Section 5: Global Space-Time Tradeoff & Range Dominance

This subproject implements and reproduces the experiments, mathematical theorems, and visual results for **Section 5: Global Space-Time Tradeoff & Range Dominance** from the foundational paper:
> **"Bitmap Index Design and Evaluation"** — Chee-Yong Chan & Yannis E. Ioannidis (ACM SIGMOD 1998).

---

## 📁 Directory Structure

```text
global_space_time_frontier_section5/
├── README.md
├── code/
│   ├── global_design_space_frontier_simulation.py
└── results/
    ├── figure9a_global_frontier_cardinality20.png
    ├── figure9b_global_frontier_cardinality100.png
    ├── figure9c_global_frontier_cardinality1000.png
    ├── figure9_all_cardinalities_frontier_grid.png
    ├── range_vs_equality_dominance_proof.png
    ├── empirical_overlay_on_theoretical_frontier_C20.png
    ├── pareto_frontier_cardinality20.csv
    ├── pareto_frontier_cardinality100.csv
    ├── pareto_frontier_cardinality1000.csv
```

---

## 🚀 Execution

To execute the simulation and regenerate all output plots and datasets:

```bash
python code/global_space_time_frontier_section5/code/global_design_space_frontier_simulation.py
```

---

## 📊 Outputs & Artifacts

### Scripts in `code/`:
- `code/global_design_space_frontier_simulation.py`: Main executable script

### Artifacts in `results/`:
- `results/figure9a_global_frontier_cardinality20.png`
- `results/figure9b_global_frontier_cardinality100.png`
- `results/figure9c_global_frontier_cardinality1000.png`
- `results/figure9_all_cardinalities_frontier_grid.png`
- `results/range_vs_equality_dominance_proof.png`
- `results/empirical_overlay_on_theoretical_frontier_C20.png`
- `results/pareto_frontier_cardinality20.csv`
- `results/pareto_frontier_cardinality100.csv`
- `results/pareto_frontier_cardinality1000.csv`
