# Saxpy Kokkkos SIMD

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
|Real-word Aproximation Memory Bandwidth| ~70 GB/s (requires check)|



### cpuinfo flags


- lm: Long Mode -> 64-bit architecture
- smx: Safer Mode Extensions -> chipset that provides enforcement of protection mechanisms
- pae: Physical Address Extension -> allows our CPUs to access physical memory sizes greater than 4 GB
- acpi: Advanced Configuration and Power Interface -> discover and configure computer hardware components, to perform power management (e.g. putting unused hardware components to sleep), auto configuration (e.g. plug and play and hot swapping), and status monitoring.
- sse: Streaming SIMD Extension -> allows for SIMD
- sse2: Unlike SSE, SSE2 is capable of handling 64-bit value. +144 instructions
- sse3: +13 new instructions
- sse4_1 and sse4_2: HD Boost -> contain subsets of 54 new instructions.
- ht: Hyper-Threading -> Only on P-cores (CPU0/1, CPU2/3).
- tm + tm2: Thermal Monitor -> reduces its thermal output by reducing its clock speed
- pdcm: Perfomance and Debugging Capabilities MSR (Model-Specific Register) -> debugging and benchmarks



## CPU Base Frequency and Thread Siblings
```bash
for cpu_path in /sys/devices/system/cpu/cpu[0-9]*; do
  cpu_num=$(basename "$cpu_path" | sed 's/cpu//');
  base_freq=$(cat "$cpu_path/cpufreq/base_frequency" 2>/dev/null || echo "N/A");
  siblings=$(cat "$cpu_path/topology/thread_siblings_list" 2>/dev/null || echo "N/A");
  echo "CPU$cpu_num: Base Freq = $base_freq MHz, Siblings = $siblings";
done

```
CPU0: Base Freq = 1300 MHz, Siblings = 0-1, Type = P-core (Hyper-Threading)
CPU1: Base Freq = 1300 MHz, Siblings = 0-1, Type = P-core (Hyper-Threading)
CPU10: Base Freq = 800 MHz, Siblings = 10, Type = E-core
CPU11: Base Freq = 800 MHz, Siblings = 11, Type = E-core
CPU12: Base Freq = 700 MHz, Siblings = 12, Type = LP E-core
CPU13: Base Freq = 700 MHz, Siblings = 13, Type = LP E-core
CPU2: Base Freq = 1300 MHz, Siblings = 2-3, Type = P-core (Hyper-Threading)
CPU3: Base Freq = 1300 MHz, Siblings = 2-3, Type = P-core (Hyper-Threading)
CPU4: Base Freq = 800 MHz, Siblings = 4, Type = E-core
CPU5: Base Freq = 800 MHz, Siblings = 5, Type = E-core
CPU6: Base Freq = 800 MHz, Siblings = 6, Type = E-core
CPU7: Base Freq = 800 MHz, Siblings = 7, Type = E-core
CPU8: Base Freq = 800 MHz, Siblings = 8, Type = E-core
CPU9: Base Freq = 800 MHz, Siblings = 9, Type = E-core

- P-cores: 1.3 GHz, with Hyper-Threading (CPU0/1, CPU2/3).
- E-cores: 0.8 GHz, no Hyper-Threading (CPU4–CPU11).
- LP E-cores: 0.7 GHz, no Hyper-Threading (CPU12–CPU13).


## Saxpy

### Kernal

`y[i] = a * x[i] + y[i]`

- 2 FLOPS per element
- 3 Memory accesses per element (read `x[i]`, read/write `y[i]`)
- data size per element 4 bytes (float / int)

### Operational intensity

$$
I = \frac{W}{Q}
$$
> Where:  
> $W = 2$ : work in FLOPS  
> $Q = 3 * 4$ bytes : memory traffic in bytes

$$
I = \frac{2}{3 \cdot 4}  = 0.167  \ \text{FLOPS/byte}
$$

### Theoretical Minimum execution time of kernel

- if we consider it is perfectly optimized and only bound on memory bandwidth:
  - total data transferred: $D =N \cdot 3 \cdot 4 \ \text{bytes} = 12 \cdot N \ \text{bytes}$ 
  - bandwidth: $B  \approx 70 \ \text{GB/s}$

$$
\boxed{
T_{theoreictal} = \frac{D}{B}
}
$$


## STREAM Triad

- STREAM Triad bench mark matches very closely to the SAXPY kernel
- compiled with (single threaded): 
```bash
gcc -O2 -fno-tree-vectorize stream.c -o stream
```


-------------------------------------------------------------
Function    Best Rate MB/s  Avg time     Min time     Max time
Copy:           31109.2     0.005208     0.005143     0.005348
Scale:          18331.7     0.008783     0.008728     0.008887
Add:            21471.2     0.011314     0.011178     0.012051
Triad:          20934.9     0.011502     0.011464     0.011589
-------------------------------------------------------------


### Scalar prediction

estimated single core memory bandwidth

For $N=10 \times 10 ^ {5} = 100000  $:

- total bytes = $3 \cdot 4 \cdot N = 0.012 \ \text{GB}$
- total time:
$$
T_{total} =\frac{1.2  \ \text{GB}}{6.4 \ \text{GB/s}}    = 0.0018 \ \text{s}
$$

