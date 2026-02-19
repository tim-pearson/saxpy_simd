# SAXPY Benchmark with Kokkos Scalar, Kokkos SIMD and Base C++ 

This project benchmarks the **SAXPY** operation (`y = a * x + y`) using different implementations:

1. **Scalar Kokkos** – a simple Kokkos `parallel_for` with a serial loop.
2. **SIMD Kokkos** – a Kokkos `parallel_for` leveraging SIMD vectorization.
3. **Scalar Base** – a standard C++ loop without Kokkos.

The goal is to compare **performance** across these methods and observe the effects of compiler optimization flags(all results are compiled with the `-fno-tree-vectorize` flag
):
- no optimization flag
- `-O2` flag
- `-O3` flag


The benchmarks record execution time for varying problem sizes (50,000 -> 1,000,000), and the results are presented as:

* **Plots** – showing time vs. problem size for each implementation using `gnuplot`.
* **Tables** – including speedups


By evaluating these configurations across increasing problem sizes, this project highlights the trade-offs between abstraction overhead, explicit SIMD
vectorization, and compiler-driven optimizations in a simple but representative numerical kernel.



## System Info

[https://www.techpowerup.com/cpu-specs/core-ultra-5-125u.c3557]

| Vendor ID: |                   GenuineIntel |
| ---------- | ------------------------------ |
|Model name  |                Intel(R) Core(TM) Ultra 5 125U |
|CPU family|              6 |
|Model|                   170 |
|Thread(s) per core|      2 |
|Core(s) per socket|      12 |
|Socket(s)|               1 |
|Stepping|                4 |
|CPU(s) scaling MHz|      54% |
|CPU max MHz|             4300.0000 |
|CPU min MHz|             400.0000 |
|BogoMIPS|                5376.00
|Theoretical Memory Bandwidth| 89.6 GB/s|




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



### CPU Base Frequency and Thread Siblings
```bash
for cpu_path in /sys/devices/system/cpu/cpu[0-9]*; do
  cpu_num=$(basename "$cpu_path" | sed 's/cpu//');
  base_freq=$(cat "$cpu_path/cpufreq/base_frequency" 2>/dev/null || echo "N/A");
  siblings=$(cat "$cpu_path/topology/thread_siblings_list" 2>/dev/null || echo "N/A");
  echo "CPU$cpu_num: Base Freq = $base_freq MHz, Siblings = $siblings";
done

```
- P-cores: 1.3 GHz, with Hyper-Threading.
- E-cores: 0.8 GHz, no Hyper-Threading.
- LP E-cores: 0.7 GHz, no Hyper-Threading.

## Prediction

> we will use the namespace `namespace KE = Kokkos::Experimental;` 

With this specific cpu, the value of `KE::simd_size` given we have initialized with `KE::simd<int` is **8**.
We can expect that there will be a speed up of ~x8.
This value is expected to drop when building with the `-O2` flag and further with the `-O3` flag.

