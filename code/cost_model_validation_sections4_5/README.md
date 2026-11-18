# Sections 4 & 5: Closed-Form Cost Model Validation (Equations 1-4)

This subproject implements and reproduces the experiments, mathematical theorems, and visual results for **Sections 4 & 5: Closed-Form Cost Model Validation (Equations 1-4)** from the foundational paper:
> **"Bitmap Index Design and Evaluation"** — Chee-Yong Chan & Yannis E. Ioannidis (ACM SIGMOD 1998).

---

## 📁 Directory Structure

```text
cost_model_validation_sections4_5/
├── README.md
├── code/
│   ├── analytical_cost_model_validation.py
└── results/
    ├── space_cost_model_validation.png
    ├── range_query_time_model_validation.png
    ├── equality_query_time_model_validation.png
    ├── prediction_error_distribution_histogram.png
    ├── cost_model_validation_metrics.csv
```

---

## 🚀 Execution

To execute the simulation and regenerate all output plots and datasets:

```bash
python code/cost_model_validation_sections4_5/code/analytical_cost_model_validation.py
```

---

## 📊 Outputs & Artifacts

### Scripts in `code/`:
- `code/analytical_cost_model_validation.py`: Main executable script

### Artifacts in `results/`:
- `results/space_cost_model_validation.png`
- `results/range_query_time_model_validation.png`
- `results/equality_query_time_model_validation.png`
- `results/prediction_error_distribution_histogram.png`
- `results/cost_model_validation_metrics.csv`
