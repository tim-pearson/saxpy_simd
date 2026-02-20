# SAXPY Benchmark with Kokkos Scalar, Kokkos SIMD and Base C++ 

This project benchmarks the **SAXPY** operation (`y = a * x + y`) using different implementations:

1. **Scalar Kokkos** – a simple Kokkos `parallel_for` with a serial loop.
2. **SIMD Kokkos** – a Kokkos `parallel_for` leveraging SIMD vectorization.
3. **Scalar Base** – a standard C++ loop without Kokkos.

The goal is to compare across these three kernels, mainly:
- the overhead of the Scalar Kokkos abstraction overhead
- the performance gain between a regular C++ loop (vectorized and non vectorized)

The project is build with compiler flags:
- `-O3 -march=native -fno-tree-vectorize` 
- `-O3 -march=native -ftree-vectorize` 


The benchmarks record execution time for varying problem sizes N = (1000 -> 65,536,000) and **repeated** `REPEAT_COUNT=8` times and averaged (no warm up).





## System Info

[https://www.intel.com/content/www/us/en/products/sku/237330/intel-core-ultra-5-processor-125u-12m-cache-up-to-4-30-ghz/specifications.html]
[https://www.cpubenchmark.net/cpu.php?cpu=Intel+Core+Ultra+5+125U&id=5840]

| **Specification**                     | **Value**                          |
|--------------------------------------|------------------------------------|
| **Product Collection**               | Intel® Core™ Ultra processors (Series 1) |
| **Code Name**                        | Products formerly Meteor Lake      |
| **Vertical Segment**                 | Mobile                             |
| **Processor Number**                 | 125U                               |
| **Overall Peak TOPS (Int8)**          | 21                                 |
| **Total Cores**                      | 12                                 |
| **# of Performance-cores**           | 2                                  |
| **# of Efficient-cores**             | 8                                  |
| **# of Low Power Efficient-cores**   | 2                                  |
| **Total Threads**                    | 14                                 |
| **Max Turbo Frequency**              | 4.3 GHz                            |
| **Performance-core Max Turbo Frequency** | 4.3 GHz                        |
| **Efficient-core Max Turbo Frequency**  | 3.6 GHz                        |
| **Low Power Efficient-core Max Turbo Frequency** | 2.1 GHz      |
| **Performance-core Base Frequency**   | 1.3 GHz                            |
| **Efficient-core Base Frequency**    | 800 MHz                            |
| **Low Power Efficient-core Base Frequency** | 700 MHz               |
| **Cache**                            | 12 MB Intel® Smart Cache            |
| **Processor Base Power**             | 15 W                               |
| **Maximum Turbo Power**              | 57 W                               |
| **Minimum Assured Power**            | 12 W                               |
| **Intel® Deep Learning Boost (Intel® DL Boost) on CPU** | Yes       |
| **AI Software Frameworks Supported by CPU** | OpenVINO™, WindowsML, ONNX RT |
| **CPU Lithography**                  | Intel 4                            |
| L1 Cache |64KB|
| L2 Cache |2048KB|
| L3 Cache |12MB|




### cpuinfo flags

- lm: Long Mode -> 64-bit architecture
- smx: Safer Mode Extensions -> chipset that provides enforcement of protection mechanisms
- pae: Physical Address Extension -> allows our CPUs to access physical memory sizes greater than 4 GB
- acpi: Advanced Configuration and Power Interface -> discover and configure computer hardware components, to perform power management]
(e.g. putting unused hardware components to sleep), auto configuration (e.g. plug and play and hot swapping), and status monitoring.
- sse: Streaming SIMD Extension -> allows for SIMD
- sse2: Unlike SSE, SSE2 is capable of handling 64-bit value. +144 instructions
- sse3: +13 new instructions
- sse4_1 and sse4_2: HD Boost -> contain subsets of 54 new instructions.
- ht: Hyper-Threading -> Only on P-cores (CPU0/1, CPU2/3).
- tm + tm2: Thermal Monitor -> reduces its thermal output by reducing its clock speed
- pdcm: Perfomance and Debugging Capabilities MSR (Model-Specific Register) -> debugging and benchmarks




## Results

> Note: the Cache Levels are purely assumed for the vector size and cpu cache sizes

###  Non-vectorized `-O3 -march=native -fno-tree-vectorize`


| N        | Vector Size (Bytes) | Cache Level | scalar_kokkos | --->   | scalar_base | --->   | simd_kokkos |
| -------- | ------------------- | ----------- | ------------- | ------ | ----------- | ------ | ----------- |
| 1000     | 4000                | L1          | 5.95e-07      | x1.695 | 3.51e-07    | x1.403 | 2.5025e-07  |
| 2000     | 8000                | L1          | 8.44625e-07   | x1.29  | 6.54875e-07 | x2.363 | 2.77125e-07 |
| 4000     | 16000               | L1          | 1.55125e-06   | x1.219 | 1.27288e-06 | x3.142 | 4.05125e-07 |
| 8000     | 32000               | L1          | 2.96712e-06   | x1.174 | 2.52675e-06 | x2.615 | 9.66375e-07 |
| 16000    | 64000               | L1          | 5.76938e-06   | x1.138 | 5.07125e-06 | x2.987 | 1.69775e-06 |
| 32000    | 128000              | L2          | 1.15759e-05   | x1.151 | 1.00571e-05 | x3.168 | 3.17487e-06 |
| 64000    | 256000              | L2          | 2.25595e-05   | x1.131 | 1.9947e-05  | x3.254 | 6.13e-06    |
| 128000   | 512000              | L2          | 4.50224e-05   | x1.127 | 3.99349e-05 | x3.32  | 1.20279e-05 |
| 256000   | 1024000             | L2          | 9.76101e-05   | x1.141 | 8.55405e-05 | x2.713 | 3.15281e-05 |
| 512000   | 2048000             | L2          | 0.000210479   | x1.106 | 0.000190236 | x2.146 | 8.86278e-05 |
| 1024000  | 4096000             | L3          | 0.000427874   | x1.091 | 0.000392091 | x2.153 | 0.000182131 |
| 2048000  | 8192000             | L3          | 0.000961716   | x1.096 | 0.000877118 | x1.284 | 0.000683299 |
| 4096000  | 16384000            | RAM         | 0.00197998    | x1.123 | 0.0017625   | x1.214 | 0.00145217  |
| 8192000  | 32768000            | RAM         | 0.00409084    | x1.131 | 0.00361692  | x1.146 | 0.00315533  |
| 16384000 | 65536000            | RAM         | 0.00836405    | x1.118 | 0.00747864  | x1.148 | 0.00651543  |
| 32768000 | 131072000           | RAM         | 0.0164829     | x1.127 | 0.0146318   | x1.118 | 0.0130907   |
| 65536000 | 262144000           | RAM         | 0.0333406     | x1.126 | 0.0296062   | x1.123 | 0.0263533   |


![MISSING IMAGE](./results/no-vectorize_plot.png)  

###  Vectorized `-O3 -march=native -ftree-vectorize`


| N        | Vector Size (Bytes) | Cache Level | scalar_kokkos | --->   | scalar_base | --->   | simd_kokkos |
| -------- | ------------------- | ----------- | ------------- | ------ | ----------- | ------ | ----------- |
| 1000     | 4000                | L1          | 7.7675e-07    | x5.732 | 1.355e-07   | x0.391 | 3.46875e-07 |
| 2000     | 8000                | L1          | 1.14212e-06   | x5.582 | 2.04625e-07 | x0.542 | 3.7725e-07  |
| 4000     | 16000               | L1          | 2.166e-06     | x4.028 | 5.3775e-07  | x0.773 | 6.96e-07    |
| 8000     | 32000               | L1          | 4.06613e-06   | x3.133 | 1.298e-06   | x0.984 | 1.3195e-06  |
| 16000    | 64000               | L1          | 8.9725e-06    | x3.562 | 2.51875e-06 | x0.918 | 2.7445e-06  |
| 32000    | 128000              | L2          | 1.79194e-05   | x3.588 | 4.99437e-06 | x0.811 | 6.15763e-06 |
| 64000    | 256000              | L2          | 3.45264e-05   | x3.595 | 9.60375e-06 | x0.77  | 1.24656e-05 |
| 128000   | 512000              | L2          | 7.01261e-05   | x4.323 | 1.62219e-05 | x0.833 | 1.94785e-05 |
| 256000   | 1024000             | L2          | 0.000164492   | x3.056 | 5.38288e-05 | x0.782 | 6.88042e-05 |
| 512000   | 2048000             | L2          | 0.000212515   | x2.41  | 8.81871e-05 | x0.994 | 8.87037e-05 |
| 1024000  | 4096000             | L3          | 0.000435449   | x2.389 | 0.000182235 | x0.986 | 0.000184744 |
| 2048000  | 8192000             | L3          | 0.000966489   | x1.597 | 0.000605346 | x0.975 | 0.000620688 |
| 4096000  | 16384000            | RAM         | 0.0019907     | x1.404 | 0.00141824  | x0.978 | 0.00144998  |
| 8192000  | 32768000            | RAM         | 0.00412202    | x1.328 | 0.00310382  | x0.975 | 0.00318322  |
| 16384000 | 65536000            | RAM         | 0.00828649    | x1.293 | 0.00640749  | x0.982 | 0.0065239   |
| 32768000 | 131072000           | RAM         | 0.016772      | x1.304 | 0.0128663   | x0.979 | 0.0131478   |
| 65536000 | 262144000           | RAM         | 0.0330906     | x1.29  | 0.0256478   | x0.98  | 0.0261706   |

![MISSING IMAGE](./results/vectorize_plot.png)  
