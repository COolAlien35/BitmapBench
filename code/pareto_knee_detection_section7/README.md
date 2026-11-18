# Section 7: Pareto Knee Detection via Gradient Analysis

This subproject implements and reproduces the experiments, mathematical theorems, and visual results for **Section 7: Pareto Knee Detection via Gradient Analysis** from the foundational paper:
> **"Bitmap Index Design and Evaluation"** — Chee-Yong Chan & Yannis E. Ioannidis (ACM SIGMOD 1998).

---

## 📁 Directory Structure

```text
pareto_knee_detection_section7/
├── README.md
├── code/
│   ├── pareto_knee_gradient_analysis_simulation.py
└── results/
    ├── pareto_knee_point_identification_C100.png
    ├── left_and_right_gradients_bar_chart_C100.png
    ├── gradient_ratio_vs_component_count_C100.png
    ├── knee_point_sensitivity_across_span_scaling.png
    ├── gradient_metrics_by_component_C100.csv
    ├── pareto_knee_summary_table.csv
```

---

## 🚀 Execution

To execute the simulation and regenerate all output plots and datasets:

```bash
python code/pareto_knee_detection_section7/code/pareto_knee_gradient_analysis_simulation.py
```

---

## 📊 Outputs & Artifacts

### Scripts in `code/`:
- `code/pareto_knee_gradient_analysis_simulation.py`: Main executable script

### Artifacts in `results/`:
- `results/pareto_knee_point_identification_C100.png`
- `results/left_and_right_gradients_bar_chart_C100.png`
- `results/gradient_ratio_vs_component_count_C100.png`
- `results/knee_point_sensitivity_across_span_scaling.png`
- `results/gradient_metrics_by_component_C100.csv`
- `results/pareto_knee_summary_table.csv`
