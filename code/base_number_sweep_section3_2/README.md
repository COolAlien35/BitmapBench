# Section 3.2: Base Number Decomposition Sweep & Tradeoffs

This subproject implements and reproduces the experiments, mathematical theorems, and visual results for **Section 3.2: Base Number Decomposition Sweep & Tradeoffs** from the foundational paper:
> **"Bitmap Index Design and Evaluation"** — Chee-Yong Chan & Yannis E. Ioannidis (ACM SIGMOD 1998).

---

## 📁 Directory Structure

```text
base_number_sweep_section3_2/
├── README.md
├── code/
│   ├── base_decomposition_sweep_simulation.py
└── results/
    ├── figure8a_bitmap_scans_vs_base_number.png
    ├── figure8b_bitwise_operations_vs_base_number.png
    ├── figure8_combined_scans_and_operations.png
    ├── measured_query_latency_vs_base_number.png
    ├── analytical_scans_and_operations_sweep.csv
    ├── measured_latency_sweep.csv
```

---

## 🚀 Execution

To execute the simulation and regenerate all output plots and datasets:

```bash
python code/base_number_sweep_section3_2/code/base_decomposition_sweep_simulation.py
```

---

## 📊 Outputs & Artifacts

### Scripts in `code/`:
- `code/base_decomposition_sweep_simulation.py`: Main executable script

### Artifacts in `results/`:
- `results/figure8a_bitmap_scans_vs_base_number.png`
- `results/figure8b_bitwise_operations_vs_base_number.png`
- `results/figure8_combined_scans_and_operations.png`
- `results/measured_query_latency_vs_base_number.png`
- `results/analytical_scans_and_operations_sweep.csv`
- `results/measured_latency_sweep.csv`
