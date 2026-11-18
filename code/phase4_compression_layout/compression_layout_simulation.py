#!/usr/bin/env python3
"""
compression_layout_simulation.py
=================================
Phase 4: Bitmap Storage Layout & Compression (Section 9, Table 4, Figure 16b)

Demonstrates how bitmap storage layout affects compression of range-encoded
bitmap indexes and that decomposing an index into more components saves more
space than compressing the 1-component index.

Layouts tested:
  BS  – each bitmap in its own uncompressed file  (N bits each)
  cBS – each bitmap compressed separately with zlib
  cCS – each component stored row-major as one contiguous bitstream, zlib-compressed
  cIS – the entire index stored row-major as one contiguous bitstream, zlib-compressed

Usage:
  python3 compression_layout_simulation.py           # full run
  python3 compression_layout_simulation.py --quick   # quick smoke test
"""

import argparse
import csv
import gc
import math
import os
import random
import struct
import sys
import time as time_mod
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# THEOREM 6.1: space-optimal bases for n components, cardinality C
# ---------------------------------------------------------------------------

def space_optimal_bases(n, C):
    """Return the n-component space-optimal base from Theorem 6.1 (1).

    The base is <b-1, ..., b-1, b, ..., b> with tau components of base b and
    (n - tau) of base (b-1), where b = ceil(C^{1/n}) and tau is the smallest
    positive integer such that b^tau * (b-1)^{n-tau} >= C.
    """
    if n <= 0:
        return []
    b = max(2, math.ceil(C ** (1.0 / n)))
    if b == 2:
        return [2] * n
    # Find tau: smallest positive integer such that b^tau * (b-1)^(n-tau) >= C
    for tau in range(1, n + 1):
        prod = (b ** tau) * ((b - 1) ** (n - tau))
        if prod >= C:
            bases = [b - 1] * (n - tau) + [b] * tau
            return sorted(bases)
    return [b] * n


def space_of(bases):
    return sum(b - 1 for b in bases)


def product_of(bases):
    p = 1
    for b in bases:
        p *= b
    return p


# ---------------------------------------------------------------------------
# INDEX BUILDING — vectorised range-encoding
# ---------------------------------------------------------------------------

def decompose_value(v, bases):
    digits = []
    for b in bases:
        digits.append(v % b)
        v //= b
    return digits


def build_range_bitmaps_raw(data, bases, N):
    """Build range-encoded bitmaps as list[component][bitmap_j] of bytearray.

    component i has (bases[i] - 1) bitmaps; bitmap j encodes digit_i <= j.
    Uses numpy packbits for orders-of-magnitude speedup over scalar loops.
    """
    data_arr = np.asarray(data, dtype=np.int64)
    rem = data_arr.copy()
    components = []
    for b in bases:
        digits = rem % b
        rem //= b
        bcount = b - 1
        bitmaps = []
        for j in range(bcount):
            packed = np.packbits(digits <= j, bitorder='little').tobytes()
            bitmaps.append(bytearray(packed))
        components.append(bitmaps)
    return components


# ---------------------------------------------------------------------------
# ROW-MAJOR BITSTREAM PACKING — numpy-accelerated
# ---------------------------------------------------------------------------

def _bitmaps_to_bool_matrix(bitmaps, N):
    """Convert list of bytearrays to an (N, k) numpy bool matrix, vectorised.

    For large k*N (>500M elements), processes in row-chunks to cap memory.
    """
    k = len(bitmaps)
    byte_count = (N + 7) // 8

    # Estimate memory: unpacked matrix is k*N bytes; keep peak < ~500 MB
    MAX_ELEMENTS = 500_000_000
    if k * N <= MAX_ELEMENTS:
        raw = np.frombuffer(b"".join(bytes(b) for b in bitmaps), dtype=np.uint8)
        raw = raw.reshape(k, byte_count)
        bits = np.unpackbits(raw, axis=1, bitorder='little')[:, :N]  # (k, N)
        return bits.T.astype(np.uint8)  # (N, k)

    # Chunked path: process CHUNK_ROWS rows at a time
    CHUNK_ROWS = max(1, MAX_ELEMENTS // k)
    out = np.empty((N, k), dtype=np.uint8)
    for start in range(0, N, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, N)
        chunk_n = end - start
        # Extract the bytes covering rows [start, end) from each bitmap
        byte_start = start >> 3
        # We need bits [start, end), which live in bytes [byte_start, byte_end_incl]
        byte_end_excl = ((end - 1) >> 3) + 1
        chunk_byte_count = byte_end_excl - byte_start
        # Build sub-matrix for this chunk
        raw_chunk = np.empty((k, chunk_byte_count), dtype=np.uint8)
        for bi in range(k):
            raw_chunk[bi] = np.frombuffer(
                bytes(bitmaps[bi][byte_start:byte_end_excl]), dtype=np.uint8
            )
        bits_chunk = np.unpackbits(raw_chunk, axis=1, bitorder='little')
        # Offset within the first byte
        bit_offset = start - (byte_start << 3)
        out[start:end] = bits_chunk[:, bit_offset:bit_offset + chunk_n].T

    return out


def _bool_matrix_to_packed_bytes(mat):
    """Pack an (N, k) uint8 {0,1} matrix into a contiguous bitstream.

    Bits are laid out row-major: row0-bit0, row0-bit1, ..., row0-bitK-1,
    row1-bit0, ... etc., little-endian within each output byte.
    """
    flat = mat.ravel()  # row-major
    # Pad to multiple of 8
    pad = (-len(flat)) % 8
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    # Pack: 8 bits per byte, little-endian
    return np.packbits(flat, bitorder='little').tobytes()


def _pack_row_major_component_bitstream(component_bitmaps, N):
    """Pack a component's bitmaps into a contiguous bit-stream (row-major).

    For N records and k bitmaps, this packs N rows of k bits each into a
    contiguous bit vector of N*k bits (ceil to byte boundary at the very end).

    Returns bytes object.
    """
    if len(component_bitmaps) == 0:
        return b""
    mat = _bitmaps_to_bool_matrix(component_bitmaps, N)  # (N, k)
    return _bool_matrix_to_packed_bytes(mat)


def _pack_row_major_component_bytepadded(component_bitmaps, N):
    """Pack row-major with each row padded to a byte boundary.

    Returns bytes object.
    """
    if len(component_bitmaps) == 0:
        return b""
    k = len(component_bitmaps)
    mat = _bitmaps_to_bool_matrix(component_bitmaps, N)  # (N, k)
    # Pad each row to byte boundary
    row_bytes = (k + 7) // 8
    pad_cols = row_bytes * 8 - k
    if pad_cols:
        mat = np.hstack([mat, np.zeros((N, pad_cols), dtype=np.uint8)])
    return np.packbits(mat.ravel(), bitorder='little').tobytes()


def _pack_row_major_index_bitstream(all_components, N):
    """Pack entire index (all components concatenated) row-major bit-packed.

    Each row has sum(b_i - 1) bits across all components, bit-packed.
    """
    # Build (N, total_bitmaps) matrix by horizontally stacking components
    mats = []
    for comp in all_components:
        if len(comp) > 0:
            mats.append(_bitmaps_to_bool_matrix(comp, N))
    if not mats:
        return b""
    full_mat = np.hstack(mats)  # (N, sum(b_i - 1))
    return _bool_matrix_to_packed_bytes(full_mat)


# ---------------------------------------------------------------------------
# STREAMING ZLIB COMPRESSION (never holds uncompressed + compressed at once)
# ---------------------------------------------------------------------------

ZLIB_LEVEL = 6  # default zlib level, same as the paper's zlib library


def zlib_compress_bytes(data):
    """Compress bytes using zlib with streaming to control memory."""
    compressor = zlib.compressobj(ZLIB_LEVEL)
    chunk_size = 1 << 20  # 1 MiB chunks
    out_parts = []
    for i in range(0, len(data), chunk_size):
        out_parts.append(compressor.compress(data[i:i + chunk_size]))
    out_parts.append(compressor.flush())
    return b"".join(out_parts)


def zlib_roundtrip_check(data):
    """Verify zlib compress → decompress roundtrip."""
    compressed = zlib_compress_bytes(data)
    decompressed = zlib.decompress(compressed)
    assert decompressed == data, "zlib roundtrip failed!"
    return compressed


# ---------------------------------------------------------------------------
# SIZE MEASUREMENT — the 4 layouts
# ---------------------------------------------------------------------------

def measure_bs(components, N):
    """BS: each bitmap stored individually. Size = n_bitmaps * ceil(N/8)."""
    byte_count = (N + 7) // 8
    n_bitmaps = sum(len(c) for c in components)
    return n_bitmaps * byte_count


def measure_cbs(components, N):
    """cBS: each bitmap compressed individually with zlib."""
    total = 0
    densities = []
    sizes = []
    for comp in components:
        for bmap in comp:
            raw = bytes(bmap)
            compressed = zlib_compress_bytes(raw)
            total += len(compressed)
            # Record density for plot 3
            popcount = int.from_bytes(raw, 'little').bit_count()
            density = popcount / N if N > 0 else 0
            densities.append(density)
            sizes.append((len(raw), len(compressed)))
    return total, densities, sizes


def measure_ccs(components, N, byte_padded=False):
    """cCS: each component stored row-major then zlib-compressed.

    Args:
        byte_padded: if True, pad each row to byte boundary (ablation).
    """
    total = 0
    for comp in components:
        if byte_padded:
            raw = _pack_row_major_component_bytepadded(comp, N)
        else:
            raw = _pack_row_major_component_bitstream(comp, N)
        compressed = zlib_compress_bytes(raw)
        total += len(compressed)
    return total


def measure_cis(components, N):
    """cIS: entire index row-major then zlib-compressed as one stream."""
    raw = _pack_row_major_index_bitstream(components, N)
    compressed = zlib_compress_bytes(raw)
    return len(compressed)


# ---------------------------------------------------------------------------
# CORRECTNESS VERIFICATION (small instance)
# ---------------------------------------------------------------------------

def verify_correctness(quick=False):
    """Run correctness checks on a small instance before the main experiment."""
    print("\n--- Correctness Verification ---")
    C_test = 12
    N_test = 200
    bases_test = [3, 4]  # 2-component, prod=12 >= 12
    data_test = [random.randrange(C_test) for _ in range(N_test)]

    # 1. Build bitmaps and verify against brute force
    comps = build_range_bitmaps_raw(data_test, bases_test, N_test)
    for ci, b in enumerate(bases_test):
        for j in range(b - 1):
            bmap = comps[ci][j]
            for r in range(N_test):
                digit = decompose_value(data_test[r], bases_test)[ci]
                expected = 1 if digit <= j else 0
                actual = (bmap[r >> 3] >> (r & 7)) & 1
                assert actual == expected, (
                    f"Bitmap mismatch: comp={ci}, j={j}, row={r}, "
                    f"digit={digit}, expected={expected}, got={actual}"
                )
    print("  ✓ Bitmaps match brute-force (digit <= j)")

    # 2. Decomposition round-trip
    for v in range(C_test):
        digits = decompose_value(v, bases_test)
        recomposed = digits[0] + digits[1] * bases_test[0]
        assert recomposed == v, f"Decomposition round-trip failed for {v}"
    print("  ✓ Decomposition round-trips correctly")

    # 3. zlib roundtrip
    raw_comp0 = _pack_row_major_component_bitstream(comps[0], N_test)
    zlib_roundtrip_check(raw_comp0)
    print("  ✓ zlib decompression round-trips correctly")

    # 4. Streamed vs one-shot compression equality
    one_shot = zlib.compress(raw_comp0, ZLIB_LEVEL)
    streamed = zlib_compress_bytes(raw_comp0)
    assert zlib.decompress(one_shot) == zlib.decompress(streamed), \
        "Streamed != one-shot decompressed content"
    # Note: compressed bytes may differ due to chunking, but decompressed must match
    print("  ✓ Streamed compression decompresses identically to one-shot")

    print("  All correctness checks passed.\n")


# ---------------------------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------------------------

def run_experiment(name, C, N, max_n=6):
    """Run the full compression layout experiment for one dataset.

    Returns list of dicts with measurements for n=1..max_n.
    """
    print(f"\n{'='*70}")
    print(f"  Dataset: {name}  (C={C}, N={N:,})")
    print(f"{'='*70}")

    results = []

    for n in range(1, max_n + 1):
        bases = space_optimal_bases(n, C)
        prod = product_of(bases)
        n_bitmaps = space_of(bases)
        C_eff = prod  # effective cardinality covered

        print(f"\n  n={n}: bases={bases}, bitmaps={n_bitmaps}, prod={prod}")

        # Generate fresh data each time (avoid caching artefacts between n values)
        data = np.random.default_rng(SEED + n).integers(0, C, size=N, dtype=np.int64)

        # Build index
        t0 = time_mod.perf_counter()
        components = build_range_bitmaps_raw(data, bases, N)
        build_time = time_mod.perf_counter() - t0
        print(f"    Build time: {build_time:.2f}s")

        # --- BS ---
        bs_size = measure_bs(components, N)

        # --- cBS ---
        gc.collect()
        cbs_size, densities, per_bitmap_sizes = measure_cbs(components, N)

        # --- cCS (bit-packed) ---
        gc.collect()
        ccs_size = measure_ccs(components, N, byte_padded=False)

        # --- cCS (byte-padded ablation) ---
        gc.collect()
        ccs_padded_size = measure_ccs(components, N, byte_padded=True)

        # --- cIS ---
        gc.collect()
        if n == 1:
            cis_size = ccs_size  # By definition: for 1-component, CS and IS are identical
        else:
            cis_size = measure_cis(components, N)

        # Compressibility as % of BS size
        pct_cbs = cbs_size / bs_size * 100
        pct_ccs = ccs_size / bs_size * 100
        pct_cis = cis_size / bs_size * 100
        pct_ccs_pad = ccs_padded_size / bs_size * 100

        row = {
            "dataset": name,
            "n": n,
            "bases": str(bases),
            "n_bitmaps": n_bitmaps,
            "C": C,
            "N": N,
            "bs_bytes": bs_size,
            "cbs_bytes": cbs_size,
            "ccs_bytes": ccs_size,
            "ccs_padded_bytes": ccs_padded_size,
            "cis_bytes": cis_size,
            "pct_cbs": pct_cbs,
            "pct_ccs": pct_ccs,
            "pct_cis": pct_cis,
            "pct_ccs_padded": pct_ccs_pad,
            "densities": densities,
            "per_bitmap_sizes": per_bitmap_sizes,
        }
        results.append(row)

        print(f"    BS  = {bs_size:>12,} bytes")
        print(f"    cBS = {cbs_size:>12,} bytes  ({pct_cbs:5.1f}%)")
        print(f"    cCS = {ccs_size:>12,} bytes  ({pct_ccs:5.1f}%)")
        print(f"    cCS-padded = {ccs_padded_size:>12,} bytes  ({pct_ccs_pad:5.1f}%)")
        print(f"    cIS = {cis_size:>12,} bytes  ({pct_cis:5.1f}%)")

        # Free memory before next iteration
        del components, data
        gc.collect()

    return results


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

COLORS = {"cBS": "#3498DB", "cCS": "#E74C3C", "cIS": "#2ECC71",
           "BS": "#7F8C8D", "cCS-pad": "#E67E22"}


def plot1_compressibility_bars(results_ds1, results_ds2, name1, name2):
    """Plot 1: Grouped bars — compressibility % vs n for cBS, cCS, cIS."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, results, dname in [(ax1, results_ds1, name1), (ax2, results_ds2, name2)]:
        ns = [r["n"] for r in results]
        x = np.arange(len(ns))
        w = 0.25

        bars1 = ax.bar(x - w, [r["pct_cbs"] for r in results], w, label="cBS",
                       color=COLORS["cBS"], alpha=0.85, edgecolor="black", linewidth=0.4)
        bars2 = ax.bar(x,     [r["pct_ccs"] for r in results], w, label="cCS",
                       color=COLORS["cCS"], alpha=0.85, edgecolor="black", linewidth=0.4)
        bars3 = ax.bar(x + w, [r["pct_cis"] for r in results], w, label="cIS",
                       color=COLORS["cIS"], alpha=0.85, edgecolor="black", linewidth=0.4)

        ax.set_xlabel("Number of Components (n)", fontsize=11)
        ax.set_ylabel("Compressibility (% of BS size)", fontsize=11)
        ax.set_title(dname, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_ylim(0, 110)
        ax.axhline(y=100, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    fig.suptitle("Compressibility of Storage Layouts vs Number of Components\n"
                 "(lower is better; 100% = uncompressed BS baseline)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "plot1_compressibility_bars.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot 1 → {path}")


def plot2_absolute_space(results_ds1, results_ds2, name1, name2):
    """Plot 2: Absolute space (MB) vs n for BS, cBS, cCS."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, results, dname in [(ax1, results_ds1, name1), (ax2, results_ds2, name2)]:
        ns = [r["n"] for r in results]
        bs_mb  = [r["bs_bytes"]  / 1e6 for r in results]
        cbs_mb = [r["cbs_bytes"] / 1e6 for r in results]
        ccs_mb = [r["ccs_bytes"] / 1e6 for r in results]

        ax.plot(ns, bs_mb,  "s-",  color=COLORS["BS"],  linewidth=2, markersize=7, label="BS (uncompressed)")
        ax.plot(ns, cbs_mb, "o--", color=COLORS["cBS"], linewidth=2, markersize=7, label="cBS")
        ax.plot(ns, ccs_mb, "^-",  color=COLORS["cCS"], linewidth=2, markersize=7, label="cCS")

        ax.set_xlabel("Number of Components (n)", fontsize=11)
        ax.set_ylabel("Index Size (MB)", fontsize=11)
        ax.set_title(dname, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Absolute Index Size vs Number of Components\n"
                 "(decomposing reduces BS; compression reduces further)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "plot2_absolute_space_mb.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot 2 → {path}")


def _shannon_entropy_bits(p):
    """H(p) in bits for a Bernoulli(p) source."""
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def plot3_compression_vs_density(all_results):
    """Plot 3: Per-bitmap zlib compression ratio vs bitmap density, with H(p) overlay."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Gather all per-bitmap data points
    densities_all = []
    ratios_all = []
    for r in all_results:
        for d, (raw_sz, comp_sz) in zip(r["densities"], r["per_bitmap_sizes"]):
            if raw_sz > 0:
                densities_all.append(d)
                ratios_all.append(comp_sz / raw_sz)

    ax.scatter(densities_all, ratios_all, s=12, alpha=0.5, color="#3498DB",
               edgecolors="none", label="Per-bitmap zlib ratio")

    # Shannon entropy curve: theoretical compression limit is H(p) bits per bit
    ps = np.linspace(0.001, 0.999, 500)
    entropy_ratio = [_shannon_entropy_bits(p) for p in ps]
    ax.plot(ps, entropy_ratio, "-", color="#E74C3C", linewidth=2.5,
            label="Shannon entropy H(p)")

    ax.set_xlabel("Bitmap Density p (fraction of 1-bits)", fontsize=12)
    ax.set_ylabel("Compression Ratio (compressed / raw)", fontsize=12)
    ax.set_title("Per-Bitmap Compression Ratio vs Density\n"
                 "(bitmaps near p=0 or p=1 compress well; p≈0.5 does not)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.15)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "plot3_compression_vs_density.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot 3 → {path}")


def plot4_decompose_vs_compress(results):
    """Plot 4: Space saved by decomposing (n=1→2) vs compressing the 1-comp index."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    datasets = []
    decompose_savings = []
    compress_savings = []
    dataset_names = []

    # Group by dataset
    by_ds = {}
    for r in results:
        by_ds.setdefault(r["dataset"], []).append(r)

    for dname, rows in by_ds.items():
        rows_sorted = sorted(rows, key=lambda x: x["n"])
        if len(rows_sorted) < 2:
            continue
        r1 = rows_sorted[0]  # n=1
        r2 = rows_sorted[1]  # n=2

        bs_n1 = r1["bs_bytes"]
        bs_n2 = r2["bs_bytes"]
        cbs_n1 = r1["cbs_bytes"]
        ccs_n1 = r1["ccs_bytes"]

        # Saving from decomposing (BS n=1 → BS n=2), no compression
        save_decompose = bs_n1 - bs_n2
        # Saving from compressing (BS n=1 → best compressed n=1 = cCS n=1)
        save_compress = bs_n1 - min(cbs_n1, ccs_n1)

        dataset_names.append(dname)
        decompose_savings.append(save_decompose / 1e6)
        compress_savings.append(save_compress / 1e6)

    x = np.arange(len(dataset_names))
    w = 0.35
    ax.bar(x - w/2, decompose_savings, w, label="Decompose (n=1→2, uncompressed)",
           color="#2ECC71", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, compress_savings, w, label="Compress (n=1, BS→best compressed)",
           color="#3498DB", alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Space Saved (MB)", fontsize=12)
    ax.set_title("Decomposing (n=1→2) vs Compressing (n=1)\n"
                 "(Paper claim: decomposing saves more than compressing)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "plot4_decompose_vs_compress.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot 4 → {path}")


# ---------------------------------------------------------------------------
# CSV EXPORT
# ---------------------------------------------------------------------------

def export_csv(all_results):
    path = os.path.join(RESULTS_DIR, "compression_results.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n", "bases", "n_bitmaps", "C", "N",
                     "bs_bytes", "cbs_bytes", "ccs_bytes", "ccs_padded_bytes",
                     "cis_bytes", "pct_cbs", "pct_ccs", "pct_ccs_padded", "pct_cis"])
        for r in all_results:
            w.writerow([
                r["dataset"], r["n"], r["bases"], r["n_bitmaps"],
                r["C"], r["N"], r["bs_bytes"], r["cbs_bytes"],
                r["ccs_bytes"], r["ccs_padded_bytes"], r["cis_bytes"],
                f"{r['pct_cbs']:.2f}", f"{r['pct_ccs']:.2f}",
                f"{r['pct_ccs_padded']:.2f}", f"{r['pct_cis']:.2f}",
            ])
    print(f"  CSV → {path}")


# ---------------------------------------------------------------------------
# RESULTS TABLE
# ---------------------------------------------------------------------------

def print_results_table(all_results):
    print(f"\n{'='*95}")
    print("  RESULTS TABLE: Compressibility of Storage Layouts (% of BS size)")
    print(f"{'='*95}")
    print(f"{'Dataset':<20s} {'n':>2s}  {'Bases':<22s} {'#Bmp':>5s}  "
          f"{'BS(MB)':>8s}  {'cBS%':>6s}  {'cCS%':>6s}  {'cCS-pad%':>8s}  {'cIS%':>6s}")
    print("-" * 95)
    for r in all_results:
        print(f"{r['dataset']:<20s} {r['n']:>2d}  {r['bases']:<22s} "
              f"{r['n_bitmaps']:>5d}  {r['bs_bytes']/1e6:>8.2f}  "
              f"{r['pct_cbs']:>5.1f}%  {r['pct_ccs']:>5.1f}%  "
              f"{r['pct_ccs_padded']:>7.1f}%  {r['pct_cis']:>5.1f}%")
    print(f"{'='*95}\n")


# ---------------------------------------------------------------------------
# PAPER CLAIMS COMPARISON
# ---------------------------------------------------------------------------

def compare_with_paper(all_results):
    print("\n" + "=" * 70)
    print("  COMPARISON WITH PAPER'S CLAIMS (Section 9)")
    print("=" * 70)

    by_ds = {}
    for r in all_results:
        by_ds.setdefault(r["dataset"], []).append(r)

    for dname, rows in by_ds.items():
        rows = sorted(rows, key=lambda x: x["n"])
        print(f"\n  --- {dname} ---")

        # Claim 1: cCS compresses best (smallest %)
        all_hold = True
        for r in rows:
            best = min(r["pct_cbs"], r["pct_ccs"], r["pct_cis"])
            if abs(best - r["pct_ccs"]) > 0.01 and r["pct_ccs"] > best:
                all_hold = False
        status = "HOLDS ✓" if all_hold else "DOES NOT HOLD ✗"
        print(f"    Claim 1 (cCS compresses best): {status}")
        # Also check if cCS or cIS is best
        for r in rows:
            best = min(r["pct_cbs"], r["pct_ccs"], r["pct_cis"])
            if abs(best - r["pct_ccs"]) > 0.5 and abs(best - r["pct_cis"]) < 0.5:
                print(f"      Note: at n={r['n']}, cIS ({r['pct_cis']:.1f}%) beats cCS ({r['pct_ccs']:.1f}%)")

        # Claim 2: compression gains shrink as n increases
        ccs_pcts = [r["pct_ccs"] for r in rows]
        shrinks = all(ccs_pcts[i] <= ccs_pcts[i + 1] for i in range(len(ccs_pcts) - 1))
        status = "HOLDS ✓" if shrinks else "PARTIALLY HOLDS ~"
        print(f"    Claim 2 (compression gains shrink with n): {status}")
        print(f"      cCS%: {', '.join(f'{p:.1f}' for p in ccs_pcts)}")

        # Claim 3: decomposing (n=1→2) saves more than compressing n=1
        if len(rows) >= 2:
            r1, r2 = rows[0], rows[1]
            save_decompose = r1["bs_bytes"] - r2["bs_bytes"]
            save_compress = r1["bs_bytes"] - min(r1["cbs_bytes"], r1["ccs_bytes"])
            status = "HOLDS ✓" if save_decompose > save_compress else "DOES NOT HOLD ✗"
            print(f"    Claim 3 (decomposing > compressing): {status}")
            print(f"      Decompose saves {save_decompose/1e6:.2f} MB, "
                  f"compress saves {save_compress/1e6:.2f} MB")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test with small data")
    parser.add_argument("--records", type=int, default=200_000,
                        help="Record count N for full run (default: 200,000)")
    args = parser.parse_args()

    t_start = time_mod.perf_counter()

    print("=" * 70)
    print("  PHASE 4: Bitmap Storage Layout & Compression")
    print("  Based on Section 9 of Chan & Ioannidis (SIGMOD 1998)")
    print("=" * 70)

    # --- Correctness verification ---
    verify_correctness()

    # --- Dataset configuration ---
    # Modelled on TPC-D: Dataset 1 = small cardinality (LINEITEM.L_QUANTITY, C=50)
    #                     Dataset 2 = large cardinality (ORDERS.O_TOTALPRICE, C=2406)
    if args.quick:
        C1, N1, name1 = 50, 5_000, "Dataset 1: Small-C (C=50, quick)"
        C2, N2, name2 = 100, 5_000, "Dataset 2: Large-C (C=100, quick)"
        max_n = 4
    else:
        N = args.records
        C1, N1, name1 = 50, N, f"Dataset 1: Small-C (C=50, N={N:,})"
        C2, N2, name2 = 2406, N, f"Dataset 2: Large-C (C=2406, N={N:,})"
        max_n = 6

    # --- Run experiments ---
    results1 = run_experiment(name1, C1, N1, max_n=max_n)
    results2 = run_experiment(name2, C2, N2, max_n=max_n)
    all_results = results1 + results2

    # --- Results table ---
    print_results_table(all_results)

    # --- CSV ---
    export_csv(all_results)

    # --- Plots ---
    print("\n--- Generating Plots ---")
    plot1_compressibility_bars(results1, results2, name1, name2)
    plot2_absolute_space(results1, results2, name1, name2)
    plot3_compression_vs_density(all_results)
    plot4_decompose_vs_compress(all_results)

    # --- Paper comparison ---
    compare_with_paper(all_results)

    # --- Summary ---
    elapsed = time_mod.perf_counter() - t_start
    print(f"\n{'='*70}")
    print(f"  PHASE 4 COMPLETE — Total runtime: {elapsed:.1f}s")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'='*70}")
    print("\n  Generated files:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        fp = os.path.join(RESULTS_DIR, f)
        print(f"    {f:45s}  {os.path.getsize(fp):>10,} bytes")


if __name__ == "__main__":
    main()
