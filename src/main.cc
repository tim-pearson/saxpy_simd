#include "kernals.hh"
#include "scoped_timer.hh"
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_SIMD.hpp>
#include <impl/Kokkos_InitializeFinalize.hpp>
#include <iostream>
#include <matplot/freestanding/plot.h>
#include <matplot/matplot.h>
#include <vector>

int main(int argc, char *argv[]) {
  std::vector<double> times_simd;
  std::vector<double> times_scalar;
  std::vector<double> times_base;
  std::vector<int> n_values;
  std::vector<double> speedups;
  Kokkos::initialize(argc, argv);
  {

    int n_max = 10000000;
    int a_val = 3;
    int x_val = 3;
    int y_val = 2;
    int repeat_count = 5;

    for (int cur_N = 200; cur_N <= n_max;
         cur_N = (cur_N < 100000 ? cur_N * 2 : cur_N + 1000000)) {

      Kokkos::View<int *> x_view("x_view", cur_N);
      Kokkos::View<int *> y_view("y_view", cur_N);

      Kokkos::parallel_for(
          "init", cur_N, KOKKOS_LAMBDA(const int i) {
            y_view(i) = y_val;
            x_view(i) = x_val;
          });
      Kokkos::fence();

      double time_scalar_kokkos;
      {
        ScopedTimer timer(time_scalar_kokkos, true);
        test_scalar_kokkos(cur_N, a_val, x_view, y_view);
      }
      double time_simd_kokkos;
      {
        ScopedTimer timer(time_simd_kokkos, true);
        test_scalar_kokkos(cur_N, a_val, x_view, y_view);
      }
      double time_scalar_base;
      {
        ScopedTimer timer(time_scalar_base, false);
        test_scalar_kokkos(cur_N, a_val, x_view, y_view);
      }
    }
  }
  Kokkos::finalize();

  // std::ofstream outFile("benchmark_results.csv");
  // outFile << "N,scalar_base,scalar_kokkos,simd_kokkos,Speedup\n";
  // for (size_t i = 0; i < n_values.size(); ++i) {
  //   outFile << n_values[i] << "," << times_scalar[i] << "," << times_simd[i]
  //           << "," << speedups[i] << "\n";
  // }
  // outFile.close();

  // // Plot scalar and SIMD times
  // auto fig1 = matplot::figure();
  // matplot::semilogx(n_values, times_scalar)
  //     ->line_width(2)
  //     .display_name("Scalar");
  // matplot::hold(matplot::on);
  // matplot::semilogy(n_values,
  // times_simd)->line_width(2).display_name("SIMD"); matplot::xlabel("N");
  // matplot::ylabel("Time (s)");
  // matplot::title("Scalar vs SIMD Runtime");
  // matplot::legend();
  // matplot::grid(matplot::on);

  // // Plot speedup
  // auto fig2 = matplot::figure();
  // matplot::semilogy(n_values,
  // speedups)->line_width(2).display_name("Speedup"); matplot::xlabel("N");
  // matplot::ylabel("Speedup");
  // matplot::title("SIMD Speedup vs Scalar");
  // matplot::grid(matplot::on);

  // matplot::show();
}
