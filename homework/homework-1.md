---
title: "HW1. Sparse GEMM"
---

# Submission Deadline: 2 December 2026

# Introduction

Weight pruning is a common technique for compressing neural networks. It sets
less important weights to zero, reducing the number of stored parameters and,
potentially, the number of multiply-accumulate operations (MACs). Fewer MACs do
not automatically mean lower latency: a dense GEMM kernel still processes the
zero weights. A sparse kernel must identify and skip zero blocks to benefit
from sparsity.

In this homework, you will implement and benchmark 4×4 block-sparse GEMM on a
Raspberry Pi 4. You will apply the scalar loops, NEON SIMD, row tiling/register
reuse, and simple OpenMP parallelism practiced in the course. The objective is
to compare arithmetic savings and analytical operational intensity (OI) with
measured kernel latency.

# 4×4 Block-Sparse Matrix Multiplication (GEMM)

## 1. Problem definition

The goal is to compute the matrix multiplication

\[
C_{M\times N}=A_{M\times K}W_{K\times N},
\]

where \(A\) is the input matrix, \(W\) is the weight matrix, and \(C\) is the output matrix. All matrices contain 32-bit floating-point (FP32) values stored in **row-major order**.

The input matrix \(A\) and output matrix \(C\) are dense: every element is explicitly stored in memory. The weight matrix \(W\), however, is sparse and is provided in a **4×4 block-compressed-column (BCC) format**. Instead of storing every element of \(W\), this representation stores only the nonzero 4×4 blocks. Any block that is not stored is assumed to contain sixteen zeros.

The objective is to compute the same result as dense GEMM while avoiding unnecessary operations on the missing blocks.

## 2. How the 4×4 block-sparse format works

The matrix \(W\) is partitioned into non-overlapping 4×4 blocks. Each block contains four consecutive rows and four consecutive columns of the original matrix.

The format groups these blocks by **groups of four output columns**. Within each group, only the retained blocks are stored, ordered by increasing block-row index.

Three arrays describe the sparse matrix:

- `block_col_ptr[N/4 + 1]`: identifies the range of stored blocks belonging to each group of four output columns. For group \(g\), the corresponding blocks occupy positions

  \[
  p=\texttt{block\_col\_ptr}[g],\ldots,
  \texttt{block\_col\_ptr}[g+1]-1.
  \]

- `block_row_idx[nnzb]`: stores the block-row index associated with each retained block. If a block has index \(q\), it corresponds to rows \(4q,\ldots,4q+3\) of \(W\).

- `block_values[16 * nnzb]`: stores the values of all retained blocks consecutively. Each block contains sixteen FP32 elements in row-major order.

For a stored block at position \(p\), the element at local coordinates \((r,c)\), where \(0\leq r,c<4\), is accessed as

\[
W^{(p)}_{r,c}
=
\texttt{block\_values}[16p+4r+c].
\]

Here, \(W^{(p)}\) denotes the stored 4×4 block, rather than an individual element of the original dense matrix.

The important distinction is that **the format stores blocks by output-column group, not individual nonzero elements**. Even if some entries inside a retained block are zero, all sixteen entries are represented explicitly.

## 3. How the sparse GEMM computation works

Consider one row \(i\) of the input matrix \(A\) and one group \(g\) of four output columns. The algorithm computes the four corresponding elements of the output row \(C\) by visiting only the retained blocks in that group.

For each stored block \(p\):

1. Read its block-row index \(q=\texttt{block\_row\_idx}[p]\).
2. Select the four input values corresponding to that block-row:

   \[
   a_r=A_{i,\,4q+r},\qquad r=0,1,2,3.
   \]

3. Load the sixteen weight values belonging to the stored block.
4. Multiply the four input values by the corresponding rows of the 4×4 weight block.
5. Accumulate the results into the four output columns belonging to group \(g\).

More explicitly, for each local output column \(c\), the update is

\[
C_{i,\,4g+c}
\mathrel{+}=
\sum_{r=0}^{3}
A_{i,\,4q+r}\,W^{(p)}_{r,c}.
\]

This update is performed for every retained block in the group. The input values selected by the block-row index are reused across the four output columns, while the stored weights contribute to all four output accumulators.

Because the block-row indices are ordered increasingly within each group, the algorithm visits the retained blocks in increasing order along the reduction dimension.

## 4. Why block sparsity can improve performance

A dense GEMM implementation processes every element of \(W\), including its zeros. A block-sparse implementation skips missing blocks entirely.

For each retained 4×4 block, the algorithm performs sixteen multiplications and sixteen accumulation updates per input row. A missing block requires no weight loads and no corresponding multiply-accumulate operations.

### Example: A compact BCC layout

Consider a \(W\) matrix of size \(8\times 8\), partitioned into four \(4\times4\) blocks:

\[
W=
\begin{bmatrix}
W_{00} & W_{01}\\
W_{10} & W_{11}
\end{bmatrix}.
\]

Suppose only three blocks are retained: \(W_{00}\), \(W_{10}\), and \(W_{11}\). The missing block \(W_{01}\) is treated as sixteen zeros.

<text> </text>

The block structure is:

\[
W=
\begin{bmatrix}
\boxed{W_{00}} & \boxed{0}\\
\boxed{W_{10}} & \boxed{W_{11}}
\end{bmatrix}.
\]

The BCC representation stores the blocks grouped by output-column group:

| Array | Contents | Meaning |
|---|---|---|
| `block_col_ptr` | `[0, 2, 3]` | Group 0 contains blocks at positions 0–1; group 1 contains the block at position 2. |
| `block_row_idx` | `[0, 1, 1]` | The stored blocks belong to block rows 0, 1, and 1, respectively. |
| `block_values` | `B00, B10, B11` | The three blocks, each stored as 16 consecutive FP32 values in row-major order. |

Here, `B00`, `B10`, and `B11` denote the sixteen-element sequences representing the corresponding blocks; they are labels for illustration, not literal array entries.

For example, the first block is stored as

\[
\begin{bmatrix}
w_{00} & w_{01} & w_{02} & w_{03}\\
w_{10} & w_{11} & w_{12} & w_{13}\\
w_{20} & w_{21} & w_{22} & w_{23}\\
w_{30} & w_{31} & w_{32} & w_{33}
\end{bmatrix}
\longrightarrow
[w_{00},w_{01},w_{02},w_{03},\ldots,w_{33}].
\]

The values are flattened row by row.

**How to interpret the arrays:** `block_col_ptr` determines which blocks belong to an output-column group, while `block_row_idx` determines which four rows of \(W\), and therefore which four elements of an input row of \(A\), are used for each block.

For instance, for output-column group 1, the range is

\[
p=\texttt{block\_col\_ptr}[1]
\;\text{to}\;
\texttt{block\_col\_ptr}[2]-1
=2\;\text{to}\;2.
\]

Thus, the group contains only block \(W_{11}\), whose block-row index is 1. The kernel uses input elements \(A_{i,4},\ldots,A_{i,7}\) to update output elements \(C_{i,4},\ldots,C_{i,7}\). The missing block \(W_{01}\) is never loaded or processed.

In general, `block_col_ptr` has one more entry than the number of output-column groups. Its final entry equals `nnzb`, the total number of retained blocks.

# Test case

Consider only one fixed test case with:

$$
M=128,\qquad K=N=512,\qquad nnzb=4096.
$$

and 75% of zero blocks. 

# Goal

Implement and compare these four sparse-kernel versions:

1. **NEON (`sparse_neon.c`)** — use four FP32 NEON lanes to update four
   adjacent output columns.
2. **Four-row tiled (`sparse_tiled.c`)** — compute exactly four input rows
   together and reuse each loaded weight block across those rows; do not use
   NEON in this version.
3. **OpenMP (`sparse_openmp.c`)** — parallelize independent output-row work (no tiling, no NEON in this version).
4. **Combined (`sparse_openmp_tiled_neon.c`)** — combine the four-row tile,
   four-column NEON updates, and OpenMP across independent row tiles. Use only
   a plain `#pragma omp parallel for` and test 1, 2, and 4 threads.

Do not use OpenMP clauses such as `collapse`, `schedule`, or `reduction`. Keep
the supplied function interfaces and sparse representation unchanged.

# Starter files

Download the [starter archive](https://raw.githubusercontent.com/ValentinoPeluso/ECAI27/main/assets/downloads/hw1.zip).

```bash
wget https://raw.githubusercontent.com/ValentinoPeluso/ECAI27/main/assets/downloads/hw1.zip
unzip hw1.zip
cd hw1
code .
```

The package provides:

- `src/sparse_gemm/case_data.c`: the fixed input and packed-weight fixture.
- `src/sparse_gemm/dense_gemm.c`: two working dense references, `dense_scalar`
  and `dense_tiled_neon_openmp_TN8_T4`. Use these as correctness and performance
  baselines; do not modify them.
- `src/sparse_gemm/sparse_scalar.c`: the working, single-threaded plain-C
  sparse baseline.
- `src/sparse_gemm/sparse_neon.c`, `sparse_tiled.c`, `sparse_openmp.c`, and
  `sparse_openmp_tiled_neon.c`: the four student kernel templates.
- `src/sparse_gemm/test.c` and `src/sparse_gemm/benchmark.c`: correctness tests
  and a benchmark harness. Unimplemented kernels are reported as `SKIP` in the
  tests and as `TODO` without latency in benchmark output; these do not count as
  passing tests or valid measurements.
- `include/sparse_gemm.h` and `Makefile`: kernel interfaces and build targets.
  Keep the supplied function interfaces unchanged.
- `HW1_GroupN.tex`: fillable report template. Rename it using your group
  number before filling it in.

Run the following commands from the extracted `hw1/` directory:

```bash
make clean
make sparse_gemm
make src/sparse_gemm/test_kernel
./src/sparse_gemm/test_kernel
./src/sparse_gemm/benchmark --output results_run1.csv
```
# Exercises

Complete these tasks in order. First establish the baselines and validate the
provided scalar sparse kernel. Then implement the four optimized kernels in
the Goal section. Every optimized kernel must compile on the Pi 4, pass the
provided correctness test, and have its report section completed. Do not claim
or compare latency for a `TODO` or failed kernel.

1. **Benchmark dense baselines and validate `sparse_scalar`.** Run the tests
   and benchmark for `dense_scalar`, `dense_tiled_neon_openmp_TN8_T4`, and
   `sparse_scalar`. Verify the sparse result against the dense reference.
   Record the measured dense latencies and identify the faster dense baseline
   from your measurements.
2. **Implement `sparse_neon.c` (1.0 point).** Vectorize the four adjacent
   output columns using NEON. Build and pass correctness tests; complete its
   report tables and benchmark results.
3. **Implement `sparse_tiled.c` (1.5 points).** Compute four input rows per
   tile, reuse each loaded weight block for those rows, and do not use NEON.
   Build and pass correctness tests, and benchmark results.
4. **Implement `sparse_openmp.c` (1.5 points).** Parallelize independent
   output rows using `#pragma omp parallel for`. Build and pass correctness tests, and benchmark results.
5. **Implement `sparse_openmp_tiled_neon.c` (2.0 points).** Combine the
   four-row tile, NEON updates, and OpenMP over independent row tiles using
   `#pragma omp parallel for`. TBuild and pass correctness tests, and benchmark results.

# Report

Download the [LaTeX report template](https://raw.githubusercontent.com/ValentinoPeluso/ECAI27/main/assets/downloads/HW1_GroupN.tex). Rename it to
`HW1_GroupN.tex`, replacing `N` with your group number, and fill in all
requested sections. Compile the completed template to `HW1_GroupN.pdf`.

# Deliverables

Submit a zip file named `HW1_GroupN.zip` (replace `N` with your group number,
for example `HW1_Group1.zip`) containing:

- `HW1_GroupN.pdf` — your completed, compiled report.
- `sparse_neon.c` — your completed NEON kernel (1.0 point).
- `sparse_tiled.c` — your completed four-row tiled kernel without NEON
  (1.5 points).
- `sparse_openmp.c` — your completed OpenMP kernel (1.5 points).
- `sparse_openmp_tiled_neon.c` — your completed combined kernel (2.0 points).
- The raw benchmark CSV files from both runs and the generated analysis summary
  and plots.

The four kernel scores total exactly **6.0 points**. Points for a kernel are
awarded only when its implementation passes the required correctness tests and
the corresponding report section is complete in the submitted PDF.
