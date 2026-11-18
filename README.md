# BitmapBench — Bitmap Index Design and Evaluation

A benchmark, simulation, and analysis framework evaluating bitmap indexing techniques for database management systems, based on the foundational research paper:
> **"Bitmap Index Design and Evaluation"** — Chee-Yong Chan and Yannis E. Ioannidis (SIGMOD 1998)

---

## 📁 Repository & Simulation Architecture

The repository is organized by simulation and paper section according to the structure:
$$\text{code}(\text{simulation\_name\_section\_of\_paper}(\text{code}, \text{results}))$$

```text
BitmapBench/
├── README.md
├── PHASE1_SIMULATION_REPORT.md
├── PHASE2_SIMULATION_REPORT.md
├── paper/
│   └── bitmap.pdf
│
└── code/
    ├── bitmap_utils.py                         # Root re-export of shared utilities
    ├── common/
    │   ├── __init__.py
    │   └── bitmap_utils.py                     # Core bitmap engine & anti-bias controls
    │
    ├── encoding_comparison_section2/
    │   ├── README.md
    │   ├── code/
    │   │   ├── equality_vs_range_encoding_benchmark.py
    │   │   ├── value_list_multicomponent_benchmark.py
    │   │   ├── range_multicomponent_benchmark.py
    │   │   └── visualize_encoding_tradeoffs.py
    │   └── results/
    │       ├── bitmap_count_comparison.png
    │       ├── storage_footprint_comparison.png
    │       ├── query_latency_by_predicate.png
    │       ├── cache_sensitivity_hot_vs_cold.png
    │       ├── logical_operations_breakdown.png
    │       ├── space_time_tradeoff_curve.png
    │       ├── equality_vs_range_tradeoff.png
    │       ├── single_vs_multicomponent_equality.png
    │       ├── single_vs_multicomponent_range.png
    │       ├── build_times_summary.csv
    │       ├── operation_counts_by_predicate.csv
    │       └── space_consumption_summary.csv
    │
    ├── range_eval_algorithms_section3_1/
    │   ├── README.md
    │   ├── code/
    │   │   ├── rangeeval_vs_rangeeval_opt_benchmark.py
    │   │   └── cache_aware_range_evaluation_study.py
    │   └── results/
    │       ├── table1_worst_case_operations_by_query_type.csv
    │       ├── table1_worst_case_operations_chart.png
    │       ├── figure7_rangeeval_opt_execution_trace.png
    │       ├── rangeeval_vs_rangeeval_opt_ops_comparison.png
    │       ├── sweep_operations_by_component_count.csv
    │       ├── comprehensive_benchmark_metrics.csv
    │       ├── cache_aware_query_time_comparison.png
    │       ├── cache_sensitivity_hot_vs_cold.png
    │       ├── cardinality_impact_comparison.png
    │       ├── bitmap_scans_vs_base_number.png
    │       └── bitmap_operations_vs_base_number.png
    │
    ├── base_number_sweep_section3_2/
    │   ├── README.md
    │   ├── code/
    │   │   └── base_decomposition_sweep_simulation.py
    │   └── results/
    │       ├── figure8a_bitmap_scans_vs_base_number.png
    │       ├── figure8b_bitwise_operations_vs_base_number.png
    │       ├── figure8_combined_scans_and_operations.png
    │       ├── measured_query_latency_vs_base_number.png
    │       ├── analytical_scans_and_operations_sweep.csv
    │       └── measured_latency_sweep.csv
    │
    ├── cost_model_validation_sections4_5/
    │   ├── README.md
    │   ├── code/
    │   │   └── analytical_cost_model_validation.py
    │   └── results/
    │       ├── space_cost_model_validation.png
    │       ├── range_query_time_model_validation.png
    │       ├── equality_query_time_model_validation.png
    │       ├── prediction_error_distribution_histogram.png
    │       └── cost_model_validation_metrics.csv
    │
    ├── global_space_time_frontier_section5/
    │   ├── README.md
    │   ├── code/
    │   │   └── global_design_space_frontier_simulation.py
    │   └── results/
    │       ├── figure9a_global_frontier_cardinality20.png
    │       ├── figure9b_global_frontier_cardinality100.png
    │       ├── figure9c_global_frontier_cardinality1000.png
    │       ├── figure9_all_cardinalities_frontier_grid.png
    │       ├── range_vs_equality_dominance_proof.png
    │       ├── empirical_overlay_on_theoretical_frontier_C20.png
    │       ├── pareto_frontier_cardinality20.csv
    │       ├── pareto_frontier_cardinality100.csv
    │       └── pareto_frontier_cardinality1000.csv
    │
    ├── extremal_indexes_section6/
    │   ├── README.md
    │   ├── code/
    │   │   └── extremal_points_and_monotonicity_simulation.py
    │   └── results/
    │       ├── theorem6_1_space_time_monotonicity.png
    │       ├── extremal_points_on_pareto_frontier.png
    │       ├── theorem6_1_analytical_vs_empirical_verification.png
    │       └── extremal_indexes_comprehensive_data.csv
    │
    ├── three_class_tradeoff_section7/
    │   ├── README.md
    │   ├── code/
    │   │   └── three_class_tradeoff_simulation.py
    │   └── results/
    │       ├── figure10_three_class_tradeoff_cardinality100.png
    │       ├── figure11_three_class_tradeoff_cardinality1000.png
    │       ├── design_space_envelope_bounding_proof.png
    │       ├── gradient_vectors_along_frontier_cardinality100.png
    │       ├── three_class_tradeoff_data_cardinality100.csv
    │       └── three_class_tradeoff_data_cardinality1000.csv
    │
    ├── pareto_knee_detection_section7/
    │   ├── README.md
    │   ├── code/
    │   │   └── pareto_knee_gradient_analysis_simulation.py
    │   └── results/
    │       ├── pareto_knee_point_identification_C100.png
    │       ├── left_and_right_gradients_bar_chart_C100.png
    │       ├── gradient_ratio_vs_component_count_C100.png
    │       ├── knee_point_sensitivity_across_span_scaling.png
    │       ├── gradient_metrics_by_component_C100.csv
    │       └── pareto_knee_summary_table.csv
    │
    ├── analytical_knee_formula_section7/
    │   ├── README.md
    │   ├── code/
    │   │   └── analytical_knee_formula_verification.py
    │   └── results/
    │       ├── theorem7_1_formula_vs_exact_search_comparison.png
    │       ├── knee_base_components_vs_cardinality.png
    │       ├── formula_accuracy_across_cardinality_sweep.png
    │       ├── delta_asymmetry_parameter_distribution.png
    │       └── cardinality_sweep_10_to_5000_knee_validation.csv
    │
    ├── empirical_knee_benchmark_section7/
    │   ├── README.md
    │   ├── code/
    │   │   └── empirical_knee_physical_query_benchmark.py
    │   └── results/
    │       ├── measured_query_latency_across_frontier_C100.png
    │       ├── elbow_visualization_speedup_vs_space.png
    │       ├── latency_breakdown_by_query_predicate.png
    │       └── empirical_physical_benchmark_metrics.csv
    │
    └── time_optimal_constrained_design_section8/
        ├── README.md
        ├── code/
        │   └── time_optimal_index_selection_simulation.py
        └── results/
            ├── table2_optimal_vs_heuristic_reproduction.csv
            ├── figure13_heuristic_accuracy_and_scan_difference.png
            ├── figure14_query_time_vs_space_budget_tradeoff.png
            ├── optimal_vs_heuristic_cost_scatter.png
            ├── branch_and_bound_vs_heuristic_scalability.png
            └── physical_query_latency_vs_theoretical_cost.png
```

---

## 🔬 Simulations & Paper Section Mapping

| Section | Simulation Folder | Focus / Key Result |
| :---: | :--- | :--- |
| **Section 2** | `code/encoding_comparison_section2/` | Bitmap index model, equality vs range encoding, single- vs multi-component decomposition. |
| **Section 3.1** | `code/range_eval_algorithms_section3_1/` | `RangeEval` vs `RangeEval-Opt` algorithms, Table 1 worst-case ops, Figure 7 execution trace. |
| **Section 3.2** | `code/base_number_sweep_section3_2/` | Figure 8 reproduction: base number sweep, I/O scan counts vs CPU bitwise ops. |
| **Sections 4 & 5** | `code/cost_model_validation_sections4_5/` | Validation of Equations 1–4 closed-form storage and query cost models ($R^2 > 0.99$). |
| **Section 5** | `code/global_space_time_frontier_section5/` | Figure 9 reproduction: global design space frontier, proof that Range dominates Equality. |
| **Section 6** | `code/extremal_indexes_section6/` | Theorem 6.1: Point A (Base-2), Point D (Base-$C$), space/time monotonicity across components. |
| **Section 7** | `code/three_class_tradeoff_section7/` | Figures 10 & 11: Three-class envelope (Space-Optimal, Uniform, Time-Optimal classes). |
| **Section 7** | `code/pareto_knee_detection_section7/` | Gradient analysis: Left/Right gradients, span scaling factor $F$, knee identification at $n=2$. |
| **Section 7** | `code/analytical_knee_formula_section7/` | Theorem 7.1: Closed-form knee formula $b_1 \approx \lceil \sqrt{C} \rceil, b_2 \approx \lceil C/b_1 \rceil$ with $\delta$ shift (100% agreement). |
| **Section 7** | `code/empirical_knee_benchmark_section7/` | Empirical physical query benchmark: knee captures 90% speedup for 18% storage space. |
| **Section 8** | `code/time_optimal_constrained_design_section8/` | Figures 12–15 & Table 2: `TimeOptAlg` DP vs `TimeOptHeur` heuristic under space budget $M$. |

---

## 🚀 Execution Guide

Each simulation is completely self-contained. To run any simulation and regenerate its figures and datasets:

```bash
# Example 1: Run Section 2 Encoding Comparison
python3 code/encoding_comparison_section2/code/equality_vs_range_encoding_benchmark.py

# Example 2: Run Section 3.1 Algorithm Comparison (RangeEval vs RangeEval-Opt)
python3 code/range_eval_algorithms_section3_1/code/rangeeval_vs_rangeeval_opt_benchmark.py

# Example 3: Run Section 8 Time-Optimal Index Selection (Table 2 reproduction)
python3 code/time_optimal_constrained_design_section8/code/time_optimal_index_selection_simulation.py
```
Each script saves its output plots and CSV datasets directly into its sibling `results/` folder.
