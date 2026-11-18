# RangeEval vs RangeEval-Opt: Bitmap Query Evaluation Comparison

This subproject implements, evaluates, and rigorously compares the two query evaluation algorithms for range-encoded bitmap indexes presented in Section 3 of the foundational research paper:
> **"Bitmap Index Design and Evaluation"** — Chee-Yong Chan and Yannis E. Ioannidis (ACM SIGMOD 1998, Sections 3, 3.1, 3.2).

---

## 📁 Directory Structure

```text
code/
└── range_eval_comparison/
    ├── range_eval_comparison.py        # Main simulation & benchmarking script
    ├── README.md                       # Comprehensive documentation & theory
    └── results/
        ├── benchmark_results.csv       # Measured scans, ops, and hot/cold timings
        ├── bitmap_scans_vs_base.png    # Figure 8(a): Average scans vs base number (C = 100)
        ├── bitmap_operations_vs_base.png # Figure 8(b): Average ops vs base number (C = 100)
        ├── query_time_comparison.png   # Hot cache execution latency across all 6 operators
        ├── hot_vs_cold_cache.png       # Cache sensitivity comparison (Hot vs Cold)
        └── cardinality_comparison.png  # Multi-cardinality study (C = 50, 100, 1000)
```

---

## 🧠 Algorithmic Foundation: Section 3 & Figure 6

### 1. Algorithm `RangeEval` (Original by O'Neil & Quass, 1997)
- **Concept**: Evaluates range selection queries by maintaining and updating two intermediate running bitmaps:
  - $B_{EQ}$: Tracks records that match the query value on all higher-order components.
  - $B_{LT}$ / $B_{GT}$: Accumulates records strictly less-than or greater-than the query value.
- **The Bottleneck**: To compute $B_{EQ}$ at each component $i$, it must compute an equality condition on range bitmaps:
  $$\text{digit}_i = v_i \iff B_{i, v_i} \oplus B_{i, v_i - 1}$$
  This requires $2$ bitmap scans and an XOR operation per component. Equality is the most expensive predicate in range encoding.
- **Buffering Overhead**: Requires memory buffering for at least two intermediate bitmaps ($B_{EQ}$ and $B_{LT}/B_{GT}$).

### 2. Algorithm `RangeEval-Opt` (Improved by Chan & Ioannidis, 1998)
- **The Core Breakthrough**: Eliminates the intermediate equality evaluation ($B_{EQ}$) entirely for range predicates.
- **Predicate Rewriting**: All inequality queries are rewritten in terms of $\le$:
  - $A < v \iff A \le v - 1$
  - $A > v \iff \neg(A \le v)$
  - $A \ge v \iff \neg(A \le v - 1)$
- **Direct Evaluation of $\le$**: For an $n$-component base $\langle b_n, \dots, b_1 \rangle$, $A \le (v_n, \dots, v_1)$ is computed from least-significant to most-significant component:
  $$B = B_{1, v_1}$$
  $$\text{For } i = 2 \dots n: \quad B = (B \land B_{i, v_i}) \lor B_{i, v_i - 1}$$
- **Advantages**:
  - At most **1 AND** and **1 OR** per component (reduces bitmap operations by **$\sim 50\%$**).
  - Eliminates **1 bitmap scan** per range query.
  - Requires buffering only **one running bitmap** ($B$).

---

## 🔬 Experimental Setup

1. **Dataset**: $N = 100,000$ records uniformly distributed across domain $[0, C-1]$.
2. **Main Configuration**: Cardinality $C = 100$, Base $b = 10$ ($n = 2$ components, bases $\langle 10, 10 \rangle$).
3. **Correctness Verification**: All 6 comparison operators ($<, \le, >, \ge, =, \ne$) validated against ground-truth scans.
4. **Caching Methodology**:
   - **Hot Cache**: Warm-up cycles prime CPU L1/L2 caches, measuring steady-state in-memory throughput.
   - **Cold Cache**: A 32 MiB cache eviction buffer is swept before every query, forcing memory bus transactions from DRAM.
5. **Section 3.2 Study**: Sweeping base $b \in [2, C]$ across $C \in \{50, 100, 1000\}$ over all $6C$ queries.

---

## 📊 Results & Key Findings

### 1. Operations and Scans Reduction (Reproducing Figure 8)
- **Bitmap Operations**: RangeEval-Opt cuts logical operations from an average of $\sim 8\text{--}14$ operations down to $\sim 2\text{--}4$ operations (**$\sim 50\%$ reduction**).
- **Bitmap Scans**: RangeEval-Opt saves exactly 1 scan for range predicates by avoiding the extra bitmap access required for $B_{EQ}$.
- **Figure 7 Reproduction**: On predicate $A \le 864$ (3-component Base-10), RangeEval-Opt requires **4 ops and 5 scans**, compared to RangeEval's **10 ops and 6 scans**.

### 2. Execution Time Speedup ($N = 100,000$, $C = 100$, Base-10)

| Query Operator | RangeEval (Hot) | RangeEval-Opt (Hot) | Speedup Factor |
| :---: | :---: | :---: | :---: |
| **$A < v$** | $20.96\ \mu\text{s}$ | **$3.62\ \mu\text{s}$** | **$5.79\times$ faster** |
| **$A \le v$** | $21.55\ \mu\text{s}$ | **$3.61\ \mu\text{s}$** | **$5.97\times$ faster** |
| **$A > v$** | $19.94\ \mu\text{s}$ | **$9.27\ \mu\text{s}$** | **$2.15\times$ faster** |
| **$A \ge v$** | $21.42\ \mu\text{s}$ | **$9.26\ \mu\text{s}$** | **$2.31\times$ faster** |
| **$A = v$** | $20.27\ \mu\text{s}$ | **$4.95\ \mu\text{s}$** | **$4.09\times$ faster** |
| **$A \ne v$** | $28.03\ \mu\text{s}$ | **$15.47\ \mu\text{s}$** | **$1.81\times$ faster** |

### 3. Base Number Sensitivity
- As base number $b$ increases (from 2 up to $C$):
  - Number of components $n = \lceil \log_b C \rceil$ decreases.
  - Scans and operations decrease monotonically.
  - When $b = C$ ($n = 1$), range evaluation requires only 1 scan and 0 operations.

### 4. Cardinality Scaling ($C = 50, 100, 1000$)
- As $C$ increases from 50 to 1000, the maximum number of components increases from $\sim 6$ to $10$.
- RangeEval-Opt consistently maintains its $\sim 50\%$ operations advantage across all scales.

---

## 🚀 How to Run

```bash
# Navigate to the experiment directory
cd code/range_eval_comparison

# Run the simulation and generate all plots + CSV
python3 range_eval_comparison.py

# View generated outputs
ls -l results/
```

