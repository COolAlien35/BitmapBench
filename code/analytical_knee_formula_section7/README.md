# Section 7: Analytical Knee Formula Verification (Theorem 7.1)

This subproject implements and reproduces the experiments, mathematical theorems, and visual results for **Section 7: Analytical Knee Formula Verification (Theorem 7.1)** from the foundational paper:
> **"Bitmap Index Design and Evaluation"** — Chee-Yong Chan & Yannis E. Ioannidis (ACM SIGMOD 1998).

---

## 📁 Directory Structure

```text
analytical_knee_formula_section7/
├── README.md
├── code/
│   ├── analytical_knee_formula_verification.py
└── results/
    ├── theorem7_1_formula_vs_exact_search_comparison.png
    ├── knee_base_components_vs_cardinality.png
    ├── formula_accuracy_across_cardinality_sweep.png
    ├── delta_asymmetry_parameter_distribution.png
    ├── cardinality_sweep_10_to_5000_knee_validation.csv
```

---

## 🚀 Execution

To execute the simulation and regenerate all output plots and datasets:

```bash
python code/analytical_knee_formula_section7/code/analytical_knee_formula_verification.py
```

---

## 📊 Outputs & Artifacts

### Scripts in `code/`:
- `code/analytical_knee_formula_verification.py`: Main executable script

### Artifacts in `results/`:
- `results/theorem7_1_formula_vs_exact_search_comparison.png`
- `results/knee_base_components_vs_cardinality.png`
- `results/formula_accuracy_across_cardinality_sweep.png`
- `results/delta_asymmetry_parameter_distribution.png`
- `results/cardinality_sweep_10_to_5000_knee_validation.csv`
