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

# 4×4 Block-Sparse GEMM: 


## 1. Example

Consider the matrix multiplication

$$
C = AW,
$$

where the input matrix $A$ has dimensions $1 \times 8$, the weight matrix $W$ has dimensions $8 \times 8$, and the output matrix $C$ has dimensions $1 \times 8$.

Let the input be

$$
A =
\begin{bmatrix}
1 & 2 & 3 & 4 & 5 & 6 & 7 & 8
\end{bmatrix}.
$$

The weight matrix is divided into four blocks, each with dimensions $4 \times 4$:

$$
W =
\begin{bmatrix}
B_{00} & B_{01} \\
B_{10} & B_{11}
\end{bmatrix}.
$$

Suppose the blocks are

$$
B_{00} =
\begin{bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12 \\
13 & 14 & 15 & 16
\end{bmatrix},
\qquad
B_{01} =
\begin{bmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{bmatrix},
$$

$$
B_{10} =
\begin{bmatrix}
17 & 18 & 19 & 20 \\
21 & 22 & 23 & 24 \\
25 & 26 & 27 & 28 \\
29 & 30 & 31 & 32
\end{bmatrix},
B_{11} =
\begin{bmatrix}
33 & 34 & 35 & 36 \\
37 & 38 & 39 & 40 \\
41 & 42 & 43 & 44 \\
45 & 46 & 47 & 48
\end{bmatrix},
$$

The block $B_{01}$ contains only zeros. It is therefore omitted from the sparse representation.

The equivalent dense weight matrix is

$$
W =
\begin{bmatrix}
1 & 2 & 3 & 4 & 0 & 0 & 0 & 0 \\
5 & 6 & 7 & 8 & 0 & 0 & 0 & 0 \\
9 & 10 & 11 & 12 & 0 & 0 & 0 & 0 \\
13 & 14 & 15 & 16 & 0 & 0 & 0 & 0 \\
17 & 18 & 19 & 20 & 33 & 34 & 35 & 36 \\
21 & 22 & 23 & 24 & 37 & 38 & 39 & 40 \\
25 & 26 & 27 & 28 & 41 & 42 & 43 & 44 \\
29 & 30 & 31 & 32 & 45 & 46 & 47 & 48
\end{bmatrix}.
$$

Our task is to calculate all eight output elements. For output column $j$, the matrix multiplication is

$$
C_j = \sum_{k=0}^{7} A_k W_{k,j},
\qquad j = 0,1,\ldots,7.
$$

The sparse kernel performs this computation four output columns at a time, processing only the stored blocks.

## 2. Understand the output-column groups

Since each block contains four columns and the matrix has eight columns, there are two output-column groups:

- Group $g=0$ corresponds to output columns $0,1,2,3$.
- Group $g=1$ corresponds to output columns $4,5,6,7$.

The block structure is

```text
                         Weight matrix W

                     Output-column group
                       g = 0       g = 1
                     columns 0-3  columns 4-7
                   +-------------+-------------+
Block row q = 0    |     B00     |     B01     |
rows 0-3           |    STORED   |    MISSING  |
                   +-------------+-------------+
Block row q = 1    |     B10     |     B11     |
rows 4-7           |    STORED   |    STORED   |
                   +-------------+-------------+
```

Group 0 contains blocks $B_{00}$ and $B_{10}$. Both contribute to output columns $0$ through $3$.

Group 1 contains only $B_{11}$. It contributes to output columns $4$ through $7$. The missing block $B_{01}$ contributes nothing.

**A group identifies four adjacent output columns. A block-row index identifies which four elements of the input row are used to process a stored block.**

## 3. Calculate the output values by hand

### 3.1 Group 0: output columns 0–3

Group 0 contains two blocks:

- $B_{00}$ uses input elements $A_0,A_1,A_2,A_3$.
- $B_{10}$ uses input elements $A_4,A_5,A_6,A_7$.

The four output elements are obtained by summing the contributions of these two blocks:

$$
\begin{bmatrix}
C_0 & C_1 & C_2 & C_3
\end{bmatrix}
=
\begin{bmatrix}
A_0 & A_1 & A_2 & A_3
\end{bmatrix} B_{00}
+
\begin{bmatrix}
A_4 & A_5 & A_6 & A_7
\end{bmatrix} B_{10}.
$$

For output column $0$, the contribution from $B_{00}$ is

$$
\begin{aligned}
s_{00}
&= 1(1)+2(5)+3(9)+4(13)\\
&= 1+10+27+52\\
&= 90.
\end{aligned}
$$

The contribution from $B_{10}$ is

$$
\begin{aligned}
s_{10}
&= 5(17)+6(21)+7(25)+8(29)\\
&= 85+126+175+232\\
&= 618.
\end{aligned}
$$

Therefore,

$$
C_0 = s_{00}+s_{10}=90+618=708.
$$

Repeating the same calculation for the remaining output columns gives

$$
\begin{aligned}
C_1
&= \underbrace{[1(2)+2(6)+3(10)+4(14)]}_{100}\\
&\quad+\underbrace{[5(18)+6(22)+7(26)+8(30)]}_{644}\\
&=744,
\end{aligned}
$$

$$
\begin{aligned}
C_2
&= \underbrace{[1(3)+2(7)+3(11)+4(15)]}_{110}\\
&\quad+\underbrace{[5(19)+6(23)+7(27)+8(31)]}_{670}\\
&=780,
\end{aligned}
$$

$$
\begin{aligned}
C_3
&= \underbrace{[1(4)+2(8)+3(12)+4(16)]}_{120}\\
&\quad+\underbrace{[5(20)+6(24)+7(28)+8(32)]}_{696}\\
&=816.
\end{aligned}
$$

Thus, Group 0 produces

$$
\boxed{
\begin{bmatrix}
C_0 & C_1 & C_2 & C_3
\end{bmatrix}
=
\begin{bmatrix}
708 & 744 & 780 & 816
\end{bmatrix}
}
$$

### 3.2 Group 1: output columns 4–7

Group 1 contains only $B_{11}$. The contribution of the missing block $B_{01}$ is zero, so the kernel does not need to process it.

The output values are calculated using only the last four input elements:

$$
\begin{bmatrix}
C_4 & C_5 & C_6 & C_7
\end{bmatrix}
=
\begin{bmatrix}
A_4 & A_5 & A_6 & A_7
\end{bmatrix} B_{11}.
$$

For output column $4$,

$$
\begin{aligned}
C_4
&=5(33)+6(37)+7(41)+8(45)\\
&=165+222+287+360\\
&=1034.
\end{aligned}
$$

Similarly,

$$
\begin{aligned}
C_5
&=5(34)+6(38)+7(42)+8(46)\\
&=170+228+294+368\\
&=1060,
\end{aligned}
$$

$$
\begin{aligned}
C_6
&=5(35)+6(39)+7(43)+8(47)\\
&=175+234+301+376\\
&=1086,
\end{aligned}
$$

$$
\begin{aligned}
C_7
&=5(36)+6(40)+7(44)+8(48)\\
&=180+240+308+384\\
&=1112.
\end{aligned}
$$

Thus, Group 1 produces

$$
\boxed{
\begin{bmatrix}
C_4 & C_5 & C_6 & C_7
\end{bmatrix}
=
\begin{bmatrix}
1034 & 1060 & 1086 & 1112
\end{bmatrix}
}
$$

Combining the two groups, the complete output is

$$
\boxed{
C =
\begin{bmatrix}
708 & 744 & 780 & 816 & 1034 & 1060 & 1086 & 1112
\end{bmatrix}
}
$$

This is exactly the result of the equivalent dense matrix multiplication. The sparse kernel skips the missing block $B_{01}$.

## 4. How does the BCC format describe this matrix?

The sparse matrix is stored using three arrays.

```c
block_col_ptr = {0, 2, 3};
block_row_idx = {0, 1, 1};
```

The `block_values` array stores the three retained blocks consecutively.

### 4.1 `block_col_ptr`: find the stored blocks in each group

The array `block_col_ptr` identifies the range of stored-block positions belonging to each output-column group.

Our example has two output-column groups and three stored blocks. Therefore,

```c
block_col_ptr = {0, 2, 3};
```

The three entries are offsets into the compressed block list, not indices of the original matrix columns.

```text
Compressed block list

Stored position p:     0           1           2
                    +----------+----------+----------+
Stored block:       |   B00    |   B10    |   B11    |
                    +----------+----------+----------+
Output group:       | Group 0  | Group 0  | Group 1  |
                    +----------+----------+----------+
```

The first two entries delimit Group 0, while the last two delimit Group 1:

```text
block_col_ptr = {0, 2, 3}

Group 0: stored positions [0, 2)  ->  p = 0, 1
Group 1: stored positions [2, 3)  ->  p = 2
```

The interval notation $[start,end)$ includes `start` and excludes `end`.

For Group 0, the kernel executes

```c
start = w->block_col_ptr[0];  // 0
end   = w->block_col_ptr[1];  // 2
```

It therefore processes positions $p=0$ and $p=1$, corresponding to $B_{00}$ and $B_{10}$.

For Group 1, it executes

```c
start = w->block_col_ptr[1];  // 2
end   = w->block_col_ptr[2];  // 3
```

It therefore processes only position $p=2$, corresponding to $B_{11}$.

The general loop in the reference implementation is

```c
for (size_t p = w->block_col_ptr[g];
     p < w->block_col_ptr[g + 1];
     ++p) {
    // Process stored block p.
}
```

The final entry of `block_col_ptr` is the total number of stored blocks, `nnzb`. Here, that value is $3$.

### 4.2 `block_row_idx`: find the block's position in the reduction dimension

The array `block_row_idx` tells the kernel which block row of $W$ each stored block occupies:

```c
block_row_idx = {0, 1, 1};
```

| Stored position `p` | `block_row_idx[p]` | Block | Rows of \(W\) |
|---:|---:|---|---|
| 0 | 0 | \(B_{00}\) | 0–3 |
| 1 | 1 | \(B_{10}\) | 4–7 |
| 2 | 1 | \(B_{11}\) | 4–7 |

The block-row index is a zero-based index. Each increment corresponds to four rows of the original matrix.

If the block-row index is $q$, the first row of that block in $W$ is

$$
k_0 = 4q.
$$

The four reduction indices are then

$$
k = k_0+r,
\qquad r=0,1,2,3.
$$

This is implemented in C as

```c
size_t k0 = 4 * (size_t)w->block_row_idx[p];
```

For example, when `p = 1`,

```c
w->block_row_idx[1] == 1
```

so `k0 = 4`. The kernel uses input indices $4,5,6,7$.

When `p = 0`, the block-row index is zero, so `k0 = 0` and the kernel uses input indices $0,1,2,3$.

There is no `block_row_ptr` in this format. `block_col_ptr` locates the list of blocks for each output group, while `block_row_idx` tells the kernel where each block belongs vertically.

### 4.3 `block_values`: find the sixteen weights in a block

Each stored block occupies sixteen consecutive entries in `block_values`. The elements of a block are stored row by row.

For our example, the layout is

```text
block_values

Indices            Stored block       Values
------------------------------------------------
0 ... 15            B00                 1 ... 16
16 ... 31           B10                17 ... 32
32 ... 47           B11                33 ... 48
```

For a stored block at position $p$, its element at local row $r$ and local column `col` is

$$
\text{weight index} = 16p+4r+\text{col}.
$$

Both $r$ and `col` range from $0$ to $3$.

For example, consider element $B_{00}[2,1]$, whose value is $10$. Its block position is $p=0$, its local row is $r=2$, and its local column is $\text{col}=1$.

The element's array index is

$$
16(0)+4(2)+1=9.
$$

Therefore, `block_values[9]` contains the value $10$.

For $B_{11}[2,1]$, the block position is $p=2$:

$$
16(2)+4(2)+1=41.
$$

Therefore, `block_values[41]` contains $42$.

The corresponding C expression is

```c
w->block_values[16 * p + 4 * r + col]
```

## 5. Reference implementation

The core loop of `gemm_sparse_scalar` is

```c
for (size_t i = 0; i < m; ++i)
    for (size_t g = 0; g < w->n / 4; ++g) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (size_t p = w->block_col_ptr[g];
             p < w->block_col_ptr[g + 1]; ++p) {
            size_t k0 = 4 * (size_t)w->block_row_idx[p];

            for (size_t r = 0; r < 4; ++r) {
                float av = a[i * w->k + k0 + r];

                for (size_t col = 0; col < 4; ++col)
                    acc[col] += av *
                        w->block_values[16 * p + 4 * r + col];
            }
        }

        for (size_t col = 0; col < 4; ++col)
            c[i * w->n + 4 * g + col] = acc[col];
    }
```

The four indices have distinct roles:

| Variable | Meaning |
|---|---|
| `i` | Selects an input row. |
| `g` | Selects an output-column group. |
| `p` | Selects a stored block in the compressed block list. |
| `r` | Selects a local row within the current block. |
| `col` | Selects a local output column within the current block. |

### 5.1 Compute the global input index

The input matrix is stored in row-major order. For input row `i`, the starting address is `a + i * w->k`.

The selected block has block-row index $q$, so its first reduction index is $k_0=4q$. Local row `r` gives the global input index

$$
k = 4q+r.
$$

Thus, the C expression

```c
float av = a[i * w->k + k0 + r];
```

loads input element $A_{i,k}$.

### 5.2 Compute the global output column

Each group contains four output columns. Within the group, `col` selects one of them.

The global output column is

$$
j=4g+\text{col}.
$$

For example:

| `g` | `col` | Global output column `j` |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 0 | 2 | 2 |
| 0 | 3 | 3 |
| 1 | 0 | 4 |
| 1 | 1 | 5 |
| 1 | 2 | 6 |
| 1 | 3 | 7 |

This explains the output write:

```c
c[i * w->n + 4 * g + col] = acc[col];
```

### 5.3 Compute one multiply-accumulate update

The central statement is

```c
acc[col] += av *
    w->block_values[16 * p + 4 * r + col];
```

It multiplies one input value by a weight and adds the product to the corresponding output accumulator.

For example, consider Group 0, block position `p = 0`, local row `r = 2`, and local column `col = 1`.

The block-row index is

$$
q=\texttt{block\_row\_idx}[0]=0.
$$

The input index is

$$
k=4q+r=4(0)+2=2.
$$

The input value is

$$
A_2=3.
$$

The weight index is

$$
16p+4r+\text{col}
=16(0)+4(2)+1
=9.
$$

The corresponding weight is

$$
\texttt{block\_values}[9]=10.
$$

Therefore, this iteration updates the accumulator for output column 1 by

$$
\texttt{acc}[1]\mathrel{+}=3(10)=30.
$$

This is just one contribution. The kernel repeats the operation for every local row and every stored block in the group, accumulating all contributions before writing the four output values.

## 6. Summary

The BCC representation separates three kinds of information:

| Array | Purpose |
|---|---|
| `block_col_ptr` | Identifies the range of stored blocks for each output-column group. |
| `block_row_idx` | Identifies the block row occupied by each stored block. |
| `block_values` | Stores the sixteen FP32 values of each retained block. |

The computation follows these steps:

1. Select an output row and an output-column group.
2. Use `block_col_ptr` to find the stored blocks belonging to the group.
3. Use `block_row_idx` to locate each block along the reduction dimension.
4. Use `block_values` to retrieve the weights.
5. Multiply the weights by the corresponding input values and accumulate four output values.
6. Write the four accumulated values to the output matrix.

The optimized kernel must preserve this computation while improving data reuse, SIMD utilization, or parallel execution. The provided sparse weights and their corresponding dense representation must be used without repacking or changing the sparsity pattern.

# Test case

Consider only one fixed test case with:

$$
M=128,\qquad K=N=512,\qquad nnzb=4096.
$$


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
