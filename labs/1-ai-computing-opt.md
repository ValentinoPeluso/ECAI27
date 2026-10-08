---
title: "1. AI Computing Optimization"
---

:::{.callout-note}
We will work on this lab assignment during both the **8 October** and **22 October** sessions.  
You're expected to complete all excercises before the **29 October**, in which *Homework #1* will be assigned.
:::

## Learning objectives

- Learn and understand the main optimizations for processing AI layers efficiently, such as SIMD, parallelism, data reuse, and tiling.
- Use the roofline model to reason about memory-bound and compute-bound workloads.
- Understand the characteristics of underlying hardware (SIMD units, instruction set, memory hierarchy, multi-core etc.) and learn how to exploit them to maximize performance.
- Measure the effects on performance of different optimizations.

## Introduction

In PyTorch, neural layers are provided as high-level functions or classes. For instance, matrix-to-vector multiplication and matrix multiplication can be executed with the `torch.matmul` function. However, these functions are not implemented with Python code, otherwise they would be highly inefficient. Instead, the high-level Python API dispatches the computation to optimized low-level
implementations, called **micro-kernels**. A micro-kernel is a small, specialized piece of computation designed to maximize use of the hardware execution resources. Different hardware architectures may require different implementations of the same mathematical operation. The PyTorch runtime therefore selects an implementation appropriate for the target processor.

Schematically:

```text
                     PyTorch
                        │
                  torch.matmul
                        │
                 operator dispatch
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
    Intel CPU         ARM CPU          CUDA
      AVX2             NEON            GPU
        │               │               │
        ▼               ▼               ▼
   C/C++           C/C++               CUDA
   or assembly     or assembly
```

As you can see from the scheme above, the language in which micro-kernels are written may also change: typically C or assembly for CPU backends, CUDA for NVIDIA GPU backends.

In this lab, our target is the Raspberry Pi CPU, which is a 64-bit ARM Cortex-A72 architecture.

Micro-kernels are often manually designed following the optimizations discussed during the lectures (and many others not covered in this course). Their code is also often hand-crafted to fully exploit the available arithmetic units and memory hierarchy, using more aggressive and hardware-specific optimizations than a compiler can typically apply automatically.

## Raspberry Pi: Main Hardware Features

Neural-network workloads are computationally intensive, but many of their operations expose substantial parallelism. The Raspberry Pi CPU, the quad-core 64-bit Arm Cortex-A72, can exploit this parallelism at three complementary levels:

```text
                         Parallelism
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
         ILP                 SIMD             Multi-core
          │                   │                   │
   Independent           Multiple data        Different parts
   instructions          elements processed   of a workload
   execute concurrently  by one instruction   execute in parallel
   within one core       within one core      across cores
```

### Instruction-level Parallelism

Instruction-level parallelism exploits independent instructions within a single CPU core.
Modern CPUs like the Cortex-A72 can overlap the execution of multiple instructions through mechanisms such as pipelining. The basic idea is that, while one instruction is executing, later instructions can already be fetched/decoded, allowing concurrent processing of consecutive instructions. For example:

```text
a = b + c;
d = e + f;
g = h + i;
```

These operations are independent, so the CPU can execute parts of them concurrently inside one core.

### SIMD: the NEON Architecture

SIMD (Single Instruction, Multiple Data) allows a single instruction to operate on multiple values at once.
The Cortex-A72 cores also provide a hardware extension for SIMD processing: the NEON architecture. NEON provides 32 vector registers, each 128 bits wide. A register can therefore hold four FP32 values, and one vector instruction can process four values in parallel. For example, it is possible to process 4 additions with a single instruction:

```text
[a0 a1 a2 a3] + [b0 b1 b2 b3] -> [a0+b0 a1+b1 a2+b2 a3+b3]
```

In many cases, the compiler is able to infer operations that can be vectorized from plain C code. However, programmers can explicitly express SIMD operations using **NEON intrinsics**: hardware-specific functions with a C interface that describe vector operations and are compiled into the corresponding SIMD instructions. This is useful when the compiler does not automatically identify an effective vectorization opportunity.

Another alternative is to manually write assembly code (as often done in production-level microkernels), which however we will not cover in the labs.

In this lab, you will use only a subset of the available NEON intrinsics, as listed below:

| Intrinsic | Description |
| --- | --- |
| `vdupq_n_f32(value)` | Initialize a vector by copying `value` into all four lanes. |
| `vld1q_f32(address)` | Load four contiguous FP32 values from memory. |
| `vst1q_f32(address, value)` | Store four FP32 lanes to contiguous memory. |
| `vaddq_f32(a, b)` | Element-wise add four pairs of values. |
| `vsubq_f32(a, b)` | Element-wise subtract four pairs of values. |
| `vmulq_f32(a, b)` | Element-wise multiply four pairs of values. |
| `vfmaq_f32(acc, a, b)` | Fused multiply-add in each lane: `acc[lane] += a[lane] * b[lane]`. |
| `vaddvq_f32(value)` | Horizontally add the four FP32 lanes to produce one scalar. |

The `q` in the intrinsics denotes a 128-bit vector. The NEON instrinsic C library also provides SIMD data types: for instance, `float32x4_t` holds four FP32 values.

Example of usage:

```c
#include <arm_neon.h>

float32x4_t zero = vdupq_n_f32(0.0f); // broadcast one scalar to all 4 lanes
float32x4_t a = vld1q_f32(ptr);       // load 4 adjacent FP32 values
vst1q_f32(ptr, a);                    // store 4 FP32 values
float32x4_t s = vaddq_f32(a, b);      // add corresponding lanes
float32x4_t d = vsubq_f32(a, b);      // subtract corresponding lanes
float32x4_t p = vmulq_f32(a, b);      // multiply corresponding lanes
float32x4_t acc = vfmaq_f32(acc, a, b); // acc += a*b, lane by lane
float sum = vaddvq_f32(acc);          // add the four lanes into one scalar
```

The following example computes four element-wise products and then accumulates the results into a vector:

```c
#include <arm_neon.h>

void example(const float *a, const float *b, float *out)
{
    // Load four FP32 values from memory into 128-bit NEON vectors.
    float32x4_t va = vld1q_f32(a);
    float32x4_t vb = vld1q_f32(b);

    // Initialize all four lanes of the accumulator to zero.
    float32x4_t acc = vdupq_n_f32(0.0f);

    // Fused multiply-add:
    // acc[i] = acc[i] + va[i] * vb[i], for i = 0,...,3.
    acc = vfmaq_f32(acc, va, vb);

    // Store the four resulting FP32 values back to memory.
    vst1q_f32(out, acc);
}
```

the `vfmaq_f32` instruction computes in parallel:

```text
acc = [1×5, 2×6, 3×7, 4×8]
    = [5, 12, 21, 32]
```

The values in a four lanes can be combined into a single scalar using `vaddvq_f32`. For example:

```c
float dot_product_4(const float *a, const float *b)
{
    float32x4_t va = vld1q_f32(a);
    float32x4_t vb = vld1q_f32(b);

    float32x4_t acc = vfmaq_f32(vdupq_n_f32(0.0f), va, vb);

    return vaddvq_f32(acc);
}
```

Here, the final result is

```text
5 + 12 + 21 + 32 = 70
```

### Multi-Core Parallelism

Multi-core parallelism executes independent parts of the workload on different CPU cores.
For instance a loop such as:

```c
for (int i = 0; i < N; ++i)
    c[i] = a[i] + b[i];
```

can be split among cores, as each iteration is independent from others. The Raspberry Pi 4 CPU has four Cortex-A72 cores, so independent work can run on up to four cores.

Multi-core processing can be implemented with the **OpenMP** libary. It enables to parallelize loops by simply adding a directive `#pragma omp parallel for` before the loop, and the compiler distributes its iterations across threads. For example:

```c
#pragma omp parallel for
for (int i = 0; i < N; ++i)
    c[i] = a[i] + b[i];
```

can be split across 4 threads as follows:

```text
       c[i] = a[i] + b[i]
                │
       ┌────────┼────────┬────────┐
       ▼        ▼        ▼        ▼
     Core 0   Core 1   Core 2   Core 3
      i=0      i=1      i=2      i=3
```

Note that parallel execution has some overhead (e.g., for synchronization and communication across cores), so small loops may not become faster when split across threads.

### Memory Hierarchy

Each core has a 32 KiB data L1 cache and 48 KiB instruction L1 cache, and a 1 MiB L2 cache is shared across the four cores. The main memory is LPDDR4 SDRAM, which is used as temporary storage for  data and instructions that are currently in use. The persistend storage hosting the operating system and user data is instead supplied by an external SD card.
The memory hierarchy therefore follows a simple principle: the closer memory is to the CPU, the faster and smaller it is. A value
that is reused while it remains in a nearby cache can be accessed faster than one that must be fetched from main memory. Consequently, performance depends not only on the number of arithmetic operations, but also on how often data is moved and reused.

## Exercises Overview

In this lab, we'll cover a subset of optimizations used in micro-kernel design through a series of hands-on summarized below:

| Stage | Micro-Kernel | Variants & Optimizations |
| --- | --- | --- |
| 1 | Vector addition | Scalar baseline; SIMD instructions. |
| 2 | Sum reduction | Scalar baseline; SIMD instructions. |
| 3 | GEMV (`y = Ax`) | Scalar baseline; SIMD instructions. Data reuse. |
| 4 | GEMM (`C = AB`) | Scalar baseline; tiling and data reuse. |
| 5 | Batched GEMM | Scalar and tiled forms; multi-threading over batches or output tiles. |
| 6 | Transformer connection | Identify GEMM-like, elementwise, and reduction operations in scaled dot-product attention (discussion only). |

## 0. Preparation

### Download the starter code

You are provided with a starter code package containing:

- templates for the micro-kernels you will complete;
- compilation and build infrastructure;
- testing infrastructure;
- benchmarking infrastructure.

To download the starter code:

Download the [Lab 1 starter archive](../assets/downloads/lab1.zip), or use the terminal commands below:

1. Log in to your Raspberry Pi, either over SSH or a tunnel, and open a terminal.
2. Download and extract the starter archive, then move into the `lab1/` directory by running the commands below in the terminal:

```bash
wget https://valentinopeluso.github.io/ECAI27/assets/downloads/lab1.zip
unzip lab1.zip
cd lab1
code .
```

The starter project is organized as follows:

```text
lab1
├── Makefile               Build/test commands
├── include/kernels.h      Micro-kernel header file
├── src/
│   ├── vector_add/        Scalar and NEON vector-add kernel templates
│   ├── reduce_sum/        Scalar and NEON reduction kernel templates
│   ├── gemv/              Scalar and NEON GEMV kernel templates
│   └── gemm/              GEMM and batched-GEMM kernel templates
```

Each micro-kernel folder contains kernel source files, a test program, and a benchmark program.

For each exercise, follow the same workflow:

1. **Understand the computation and its data layout.**
2. **Implement the scalar baseline**, which will serve as the reference for correctness and performance comparisons.
3. **Build and test** the implementation.
4. **Benchmark the scalar version** and record the results.
5. **Implement the optimized version** required by the exercise.
6. **Run the correctness tests again.**
7. **Benchmark the optimized version** and compare it with the scalar baseline.
8. **Explain the observed performance** in terms of the optimization introduced, memory traffic, data reuse, and the underlying hardware.

Keep the function names, argument order, and data layouts declared in the templates.

## 1. Vector Addition

:::{.callout-note}
**Variants:** Scalar, SIMD  
**Files to complete:** `vector_add.c`, `vector_add_neon.c`
:::

### Getting started

Explore the project structure and search the vector addition templates.
The scalar version is `src/vector_add/vector_add.c`.

Open the file and complete the loop iteration code.

### Build Scalar Code

You can build the code by running the command below from the terminal:

```bash
make vector_add
```

This command mainly executes:

```bash
gcc -O0 -Wall -Wextra -std=c11 -mcpu=cortex-a72 -Iinclude -c src/vector_add/vector_add.c -o src/vector_add/vector_add.o
```

Specifically:

- `gcc`: compile C code;
- `-O0`: optimization level; disables compiler optimizations.;
- `-mcpu=cortex-a72`: tune for the Pi 4 CPU;
- `-fopenmp`: (optional) enable an OpenMP build;
- `-lm`: (optional) link the math library when needed;

### Benchmark Scalar Code

The compilation produces the benchmarking executable  `./src/vector-add/benchmark`.

The C benchmark programs use dimensions defined as constants near the start of the `main` function. To measure a different size, edit the corresponding   onstants, save the file, rebuild that target, and run it again.

For vector addition, change `n` in `src/vector_add/benchmark.c` with values of  `1024`, `16384`, and `1048576`. Tehn build a run with the commands below:

```bash
# Edit n in src/vector_add/benchmark.c, then:
make vector_add
srcvector-add/benchmark
```

Collect the reported results (e.g., in a CSV file or a Sheet) for later comparisons.

:::{.callout-warning}
**Why do we compile with `-O0`?**

In this lab, we use `-O0`, which disables compiler optimizations. Higher optimization levels such as `-O3` can provide substantial performance improvements, as the compiler will automatically apply many of the optimizations that you will study and implement yourself in this lab, as well as many others.

Using `-O0` therefore isolates the performance improvements introduced by each optimization. This allows you to measure and understand the contribution of each optimization individually.
:::

Let's see an example to explore how much optimization the compiler can perform automatically.  
Compare the same scalar implementation using `-O0` and `-O3`:

```bash
gcc -O0 -Wall -Wextra -std=c11 -mcpu=cortex-a72 -Iinclude -S src/vector_add/vector_add.c -o vector_add_O0.s
gcc -O3 -Wall -Wextra -std=c11 -mcpu=cortex-a72 -Iinclude -S src/vector_add/vector_add.c -o vector_add_O3.s
```

Compare the two assembly files and look for changes in:

- loop structure;
- instruction selection;
- register usage; and
- SIMD/vector instructions.

:::{.callout-note}
**Observation:** `-O3` enables many optimizations automatically, including transformations related to loop optimization, vectorization, instruction scheduling, and data movement. We will cover only a subset of them. In the rest of the lab, we will always use `-O0` so that you can implement and measure each introduced optimization explicitly.
:::

### Optimization

Now, we will proceed with the implementation of an optimized version using SIMD.
Complete the code of `src/vector_add/vector_add_neon.c` implementing SIMD additions using NEON intrinsics.

:::{.callout-tip}
**Handle the tail:** the number of elements may not be a multiple of 4. Process the remaining elements with a scalar `for` loop after the SIMD loop.
:::

### Test

Build and run the correctness tests.

```bash
make src/vector_add/test_kernel
./src/vector_add/test_kernel
```

### Benchmark Optimized Code

Before measuring, calculate:

- the number of floating-point operations;
- the amount of input and output data transferred; and
- the approximate operational intensity.

For vector addition,

$y_i = a_i + b_i$,

so each element requires one floating-point addition. The computation reads
two FP32 values and writes one FP32 result.

Then run the benchmarking with:

```bash
make vector_add
srcvector-add/benchmark
```

Collect the benchmarking results in the table below:

| N | Scalar time | NEON time | Speedup |
| ---: | ---: | ---: | ---: |
| 1K | | | |
| 16K | | | |
| 1M | | | |

:::{.callout-warning}
**Think about the speedup.**

NEON can process four FP32 values per instruction, but this does not guarantee a 4× end-to-end speedup. Consider the operation intensity.
:::

## 2. Sum reduction

:::{.callout-note}
**Variants:** Scalar, SIMD  
**Files to complete:** `reduce_sum.c`, `reduce_sum_neon.c`
:::

Implement a function that sums all elements of a 1D FP32 array:

```c
float reduce_sum_scalar(const float *x, size_t n);
```

### Scalar baseline

First, implement the scalar version in `reduce_sum.c`: add each element to one scalar accumulator.
For `n == 0`, return zero.

### SIMD optimization

Then, implement the SIMD version in `reduce_sum_neon.c`.

:::{.callout-tip}

- Accumulate groups of four values in a `float32x4_t` variable.  
- Aggregates the partial sums in the four lanes to a scalar.  
- Finally, handle remaining elements with a scalar tail.
:::

Test the correctness of the optimized version:

```bash
make src/reduce_sum/test_kernel
./src/reduce_sum/test_kernel
```

The test includes lengths that are shorter than a NEON vector, divisible by four, and have a SIMD tail.

:::{.callout-warning}
The validation comparisons consider an error tolerance: SIMD processing changes the order of additions, which might result in a change of the last few bits.
:::

Build and run the benchmark (scalar and SIMD):

```bash
make reduce_sum
./src/reduce_sum/benchmark
```

Given that for an input of `n` FP32 values, the reduction performs `n-1` additions and reads `4*n` bytes of input data, estimate the operational intensity and discuss the collected results.

## 3. Matrix-Vector Multiplication (GEMV)

:::{.callout-note}
**Variants:** Scalar, SIMD  
**Files to complete:** `gemv.c`, `gemv_neon.c`
:::

Matrix-vector multiplication, or **GEMV**, computes

$y = Ax$,

where

```text
A: [M,K]   x: [K]   y: [M]
```

and each output element is a dot product:

$y_i = \sum_{j=0}^{K-1} A_{i,j}x_j$.

The matrix is stored as a flat row-major array. Therefore,

```text
A[i,j] = a[i*K + j]
```

while `x[j]` is the `j`-th element of the input vector and `y[i]` is the
`i`-th output element.

### Scalar baseline

Implement `gemv_scalar` in `src/gemv/gemv.c`.

::: {.callout-tip}
For each matrix row:

1. initialize a scalar accumulator to zero;
2. multiply each matrix element by the corresponding element of `x`;
3. add the product to the accumulator; and
4. store the final sum in `y[i]`.
:::

The resulting computation is:

```text
for each row i:
    sum = 0
    for each column j:
        sum += A[i,j] * x[j]
    y[i] = sum
```

This scalar implementation is the baseline for both correctness and performance comparisons.

### SIMD optimization

Implement `gemv_neon` in `src/gemv/gemv_neon.c`.

Each output element is a dot product, which makes GEMV a natural candidate for  SIMD. Instead of processing one pair of values at a time, process four matrix/vector pairs simultaneously:

```text
A[i,j : j+3]   ×   x[j : j+3]
        ↓
  four products in parallel
        ↓
   accumulate in a
   NEON vector
        ↓
  horizontal reduction
        ↓
      y[i]
```

Use a `float32x4_t` accumulator and the NEON intrinsics introduced earlier. After processing groups of four columns, horizontally add the four lanes to obtain the scalar result for the current row.

:::{.callout-tip}
**Handle the tail:** `K` may not be a multiple of four. After the SIMD loop, process the remaining columns with a scalar loop.
:::

### Test

Check correctness:

```bash
make src/gemv/test_kernel
./src/gemv/test_kernel
```

The test uses `K=5`, deliberately leaving one element outside a complete four-element NEON group to check that the scalar tail is handled correctly.

### Benchmarking

Build and run:

```bash
make gemv
./src/gemv/benchmark
```

The benchmark uses `M=K=64`.

For GEMV, each output performs `K` multiplications and `K-1` additions. For a simple FLOP estimate, count one multiplication and one addition as two operations per matrix element:

$\mathrm{FLOPs} \approx 2MK$.

For `M=K=64`, $\mathrm{FLOPs} \approx 2\cdot64\cdot64 = 8192$.

The computation accesses:

- `M*K` FP32 values from `A`;
- `K` FP32 values from `x`; and
- `M` FP32 output values in `y`.

Estimate the operational intensity from these quantities.

An important difference from vector addition is **data reuse**. Every element of `x` is used once for every row of `A`, so the same vector values participate in many output computations. This reuse can allow `x` to remain in a nearby
cache while different rows of `A` are processed.

Complete the table below:

| Variant | Latency (µs) | GFLOP/s | Speedup vs scalar |
| --- | ---: | ---: | ---: |
| Scalar | | | 1.0× |
| NEON | | | |

:::{.callout-warning}
**Think about the speedup.**

NEON processes four FP32 values in parallel, but GEMV is not simply four times faster. Each output requires a reduction across the SIMD lanes, and the computation also moves matrix data through the memory hierarchy. Consider how these effects limit the observed speedup.
:::

## 4. Matrix-Matrix Multiplication (GEMM)

:::{.callout-note}
**Variants:** Scalar, tiled  
**Files to complete:** `gemm.c`, `gemm_tiled.c`
:::

Matrix-matrix multiplication, or **GEMM**, computes $C = AB$, where

```text
A: [M,K]   B: [K,N]   C: [M,N]
```

and

$C_{i,j} = \sum_{q=0}^{K-1} A_{i,q}B_{q,j}$.

The matrices are stored in row-major order:

```text
A[i,q] = a[i*K + q]
B[q,j] = b[q*N + j]
C[i,j] = c[i*N + j]
```

GEMM performs considerably more computation than GEMV and provides much more opportunity for **data reuse**.

### Scalar baseline

Implement `gemm_scalar` in `src/gemm/gemm.c`.

For each output element:

```text
for i:
    for j:
        sum = 0
        for q:
            sum += A[i,q] * B[q,j]
        C[i,j] = sum
```

The inner loop is a dot product between a row of `A` and a column of `B`.

For an `M × K` matrix multiplied by a `K × N` matrix, the computation requires approximately $2MKN$ floating-point operations.

### Tiling optimization

GEMM repeatedly accesses the same matrix elements while computing different output elements. **Tiling**, also called blocking, divides
the matrices into smaller submatrices so that the currently active data fits better in the cache.

Instead of considering the entire matrices at once, divide the three dimensions into blocks. For example, a tile of `C` is updated using a tile of `A` and a tile of `B`:

```text
A tile          B tile          C tile
┌─────┐         ┌─────┐         ┌─────┐
│     │    ×    │     │    →    │     │
└─────┘         └─────┘         └─────┘
```

Within a tile, values from `A` and `B` can be reused for several output elements before they are replaced by another tile. The `C` tile is updated across multiple blocks of the reduction dimension.

Implement `gemm_tiled` in `src/gemm/gemm_tiled.c` by dividing the `M`, `N`, and `K` dimensions into tiles.

:::{.callout-hint}
The matrix dimensions might not be multiples of the tile size. Clamp the end of each tile to the corresponding matrix dimension, for example:

```c
i_end = min(ii + tile, M);
```

:::

### Test

Build and run the correctness tests:

```bash
make src/gemm/test_kernel
./src/gemm/test_kernel
```

The test checks both the regular GEMM result and edge tiles whose dimensions are smaller than the selected tile size.

### Benchmarking

Build and run:

```bash
make gemm
./src/gemm/benchmark
```

The benchmark uses `M=N=K=48`.

First benchmark the scalar implementation. Then compare the tiled implementation using tile sizes `T = 4, 8, 16, 32`.

The benchmark uses a tile-size constant defined near the beginning of `src/gemm/benchmark.c`. Change this value, rebuild, and run the benchmark for each configuration.

For GEMM, we have approximately $2MKN$ FLOPs.
Estimate the data movement of the tiled implementation and compare it with the baseline implementation. Discuss how tiling increases data reuse.

Record the benchmark results:

| Variant / tile | Latency (µs) | GFLOP/s | Speedup vs scalar | Observation |
| --- | ---: | ---: | ---: | --- |
| Scalar | | | 1.0× | |
| Tiled, T=4 | | | | |
| Tiled, T=8 | | | | |
| Tiled, T=16 | | | | |
| Tiled, T=32 | | | | |

Discuss:

- what data is reused inside a tile;
- how tile size affects the working set;
- how the tile size interacts with the cache hierarchy;
- why the best tile size is not necessarily the largest one; and
- how tiling changes the balance between computation and memory traffic.

:::{.callout-caution}
A larger tile is not automatically faster. A tile that is too large may reduce
cache effectiveness, while a tile that is too small may increase loop and
indexing overhead. The best choice depends on the workload and the target
hardware.
:::

## 5. Batched Matrix-Matrix Multiplication (Batched GEMM)

:::{.callout-note}
**Variants:** Scalar, tiled, multi-core over batches, multi-core over output tiles  
**Files to complete:** `gemm_batched.c`
:::

In many AI workloads, the same matrix operation is applied independently to multiple inputs. **Batched GEMM** extends matrix-matrix multiplication to this case:

```text
A: [B,M,K]   W: [K,N]   Y: [B,M,N]

Y[b,i,j] = sum(k=0..K-1) A[b,i,k] * W[k,j]
```

Here, `B` is the batch size. Each batch element has its own input matrix `A[b]`, while all batch elements use the same weight matrix `W`.

This structure exposes two important opportunities for optimization:

- **parallelism:** different batch elements can be processed independently;
- **data reuse:** the same weight matrix `W` is reused across all batch elements.

### Serial baseline

The starter code provides a serial scalar implementation. It computes one GEMM for each batch element:

```text
for b:
    for i:
        for j:
            sum = 0
            for k:
                sum += A[b,i,k] * W[k,j]
            Y[b,i,j] = sum
```

The total computational work is approximately $2BMKN$ FLOPs.

Implement the code of `gemm_batched_scalar` in `./src/gemm/batched_gemm.c`.

Then, build and run the benchmark:

```bash
make batched_gemm_bench
./src/gemm/batched_gemm_bench
```

The benchmark uses `M=N=K=64` and batch sizes `B = 1, 4, 16, 64`

### Tiled GEMM

The tiled implementation applies the same cache-blocking strategy introduced in the previous GEMM exercise. The tile size used by the benchmark is `16`.

First, compare the serial scalar and serial tiled implementations. Consider how tiling affects the reuse of the input matrices and the shared weight matrix.

### Multi-core parallelism

Batched GEMM also provides an opportunity to parallelize execution across the four Cortex-A72 cores.

The starter code provides two parallel variants:

1. **Parallelism over batch items:** different threads process different batch elements.
2. **Parallelism over output tiles:** the output tiles of the GEMM computations are distributed across threads.

These two approaches expose parallelism in different ways.

With parallelism over batch items:

```text
             Batch
               │
       ┌───────┼───────┬───────┐
       ▼       ▼       ▼       ▼
      B=0     B=1     B=2     B=3
       │       │       │       │
     Core 0  Core 1  Core 2  Core 3
```

With parallelism over output tiles, a single batch element can provide multiple independent tasks:

```text
          Output matrix
        ┌───────┬───────┐
        │ Tile 0│ Tile 1│
        ├───────┼───────┤
        │ Tile 2│ Tile 3│
        └───────┴───────┘
             │
       independent tiles
             │
      ┌──────┼──────┬──────┐
      ▼      ▼      ▼      ▼
    Core 0 Core 1 Core 2 Core 3
```

Complete the code of the two parallel batched version.  
Then, build and run the OpenMP benchmarks:

```bash
make batched_gemm_bench_omp

OMP_NUM_THREADS=1 ./src/gemm/batched_gemm_bench_omp
OMP_NUM_THREADS=2 ./src/gemm/batched_gemm_bench_omp
OMP_NUM_THREADS=4 ./src/gemm/batched_gemm_bench_omp
```

:::{.callout-note}
**SIMD and multi-core parallelism are different.**

SIMD executes multiple data operations simultaneously within a CPU core, while OpenMP distributes independent work across multiple CPU cores. These forms of parallelism are complementary and can be combined.
:::

### Benchmarking

Compare performance of the different variants exploring `B = 1, 4, 16, 64`.
Then, compile the table below:

| B | Method | Threads | Tile | Latency (µs) | GFLOP/s | Speedup vs scalar | Observation |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Scalar | 1 | — | | | 1.0× | |
| 1 | Tiled | 1 | 16 | | | | |
| 1 | Parallel over batch | 4 | 16 | | | | |
| 1 | Parallel over tiles | 4 | 16 | | | | |
| 4 | Scalar | 1 | — | | | 1.0× | |
| 4 | Tiled | 1 | 16 | | | | |
| 4 | Parallel over batch | 4 | 16 | | | | |
| 4 | Parallel over tiles | 4 | 16 | | | | |
| 16 | Scalar | 1 | — | | | 1.0× | |
| 16 | Tiled | 1 | 16 | | | | |
| 16 | Parallel over batch | 4 | 16 | | | | |
| 16 | Parallel over tiles | 4 | 16 | | | | |
| 64 | Scalar | 1 | — | | | 1.0× | |
| 64 | Tiled | 1 | 16 | | | | |
| 64 | Parallel over batch | 4 | 16 | | | | |
| 64 | Parallel over tiles | 4 | 16 | | | | |

### Discussion

Use the measurements to answer the following questions:

- How does increasing the batch size affect the amount of available parallelism?
- How is the weight matrix `W` reused across different batch elements?
- How does tiling improve data reuse?
- Why is the measured speedup generally lower than the number of threads?

:::{.callout-warning}
**Do not expect linear scaling.**

Using four cores does not necessarily make the computation four times faster. The achievable speedup depends on the amount of parallel work, thread overhead, memory traffic, cache behavior, and the ability of the workload to keep all cores busy.
:::

## Transformer connection

Scaled dot-product attention combines the same kernel families:

```text
Attention(Q,K,V) = softmax(QK^T / sqrt(P)) V
QK^T and Attention·V  -> GEMM-like kernels
scaling                 -> elementwise operation
softmax                 -> reduction kernel
```

SIMD, parallelism, data reuse, memory traffic, tiling, and fusion are relevant to Transformer inference too. Implementing attention is outside this lab.

## Take-home message

The final lesson is: **efficient AI software is not only about reducing FLOPs; it is also about arranging computation and data so that the hardware can do useful work efficiently.**

## External Resources (optional, not needed for homework/exam)

### Videos

- [Advanced Optimizations for Matrix Multiplication](https://www.youtube.com/watch?v=6AVEPOqJfOk)
- ["Optimizing Embedded Deep Learning Inference Software," a Presentation from Arm](https://www.youtube.com/watch?v=Rv9ZmWMP3n8)
- ["Even Faster CNNs: Exploring the New Class of Winograd Algorithms," a Presentation from Arm](https://www.youtube.com/watch?v=6lvzMB56Jnc)
- [Using SGEMM and FFTs to Accelerate Deep Learning](https://www.youtube.com/watch?v=v5kAAjW17U4)

### Suggested Readings

- [The Indirect Convolution Algorithm](https://arxiv.org/abs/1907.02129)
- [Fast Sparse ConvNets](https://arxiv.org/abs/1911.09723)
- [The Two-Pass Softmax Algorithm](https://arxiv.org/abs/2001.04438)
- [High Performance and Portable Convolution Operators for ARM-based Multicore Processors](https://arxiv.org/pdf/2005.06410)
- [nDirect2: A High-Performance Library for Direct Convolutions on Multicore CPUs](https://ieeexplore.ieee.org/abstract/document/10892335/)
- [Efficient Memory Management for Deep Neural Net Inference](https://arxiv.org/abs/2001.03288)

### Micro-kernel Libraries

- [FBGEMM](https://github.com/pytorch/FBGEMM)
- [XNNPACK](https://github.com/google/xnnpack)
- [KleidiAI](https://github.com/ARM-software/kleidiai)
- [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN)
