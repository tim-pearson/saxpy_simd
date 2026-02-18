#include "kernals.hh"
#include "scoped_timer.hh"
#include <impl/Kokkos_InitializeFinalize.hpp>
#include <iomanip>
#include <matplot/freestanding/plot.h>
#include <matplot/matplot.h>
#include <vector>
//
struct Result {
  int n;
  double times[3];
};

void results_to_csv(const std::vector<Result> &results) {
  std::ofstream outFile("benchmark_results.csv");

  outFile << "N,scalar_kokkos,--->,scalar_base,--->,simd_kokkos\n";

  for (const auto &res : results) {
    double speedup_1 =
        std::round((res.times[1] / res.times[0]) * 1000.0) / 1000.0;
    double speedup_2 =
        std::round((res.times[2] / res.times[1]) * 1000.0) / 1000.0;

    outFile << res.n << "," << res.times[0] << "," << speedup_1 << ","
            << res.times[1] << "," << speedup_2 << "," << res.times[2] << "\n";
  }

  outFile.close();
}

int main(int argc, char *argv[]) {

  std::vector<Result> results;
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
        test_simd_kokkos(cur_N, a_val, x_view, y_view);
      }
      double time_scalar_base;
      {
        ScopedTimer timer(time_scalar_base, true);
        test_scalar_base(cur_N, a_val, x_view, y_view);
      }
      results.push_back(
          {.n = cur_N,
           .times = {time_scalar_kokkos, time_scalar_base, time_simd_kokkos}});
    }
  }
  Kokkos::finalize();

  results_to_csv(results);
  // std::ofstream outFile("benchmark_results.csv");
  // outFile << "N,scalar_base,scalar_kokkos,simd_kokkos\n";
  // for (auto res : results) {
  //   outFile << res.n << "," << res.times[0] << "," << res.times[1] << ","
  //           << res.times[2] << "\n";
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
