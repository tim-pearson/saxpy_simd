#include "kernals.hh"
#include "scoped_timer.hh"
#include <functional>
#include <impl/Kokkos_InitializeFinalize.hpp>
#include <matplot/freestanding/plot.h>
#include <matplot/matplot.h>
#include <vector>
#define N_MAX 10000000
#define A_VAL 3
#define X_VAL 3
#define Y_VAL 2
#define REPEAT_COUNT 5
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
        std::round((res.times[0] / res.times[1]) * 1000.0) / 1000.0;
    double speedup_2 =
        std::round((res.times[1] / res.times[2]) * 1000.0) / 1000.0;

    outFile << res.n << "," << res.times[0] << "," << speedup_1 << ","
            << res.times[1] << "," << speedup_2 << "," << res.times[2] << "\n";
  }

  outFile.close();
}

double run_test_avg(std::function<void()> test_func, bool is_kokkos) {
  int repeat_count = 5;
  double total_time = 0;
  for (int i = 0; i < 5; i++) {

    double run_time;
    {
      ScopedTimer timer(run_time, true);
      test_func();
    }
    total_time += run_time;
  }
  return total_time / repeat_count;
}

int main(int argc, char *argv[]) {

  std::vector<Result> results;
  Kokkos::initialize(argc, argv);
  {

    // int N_MAX = 10000000;
    // int A_VAL = 3;
    // int X_VAL = 3;
    // int Y_VAL = 2;
    // int REPEAT_COUNT = 5;

    for (int cur_N = 200; cur_N <= N_MAX;
         cur_N = (cur_N < 100000 ? cur_N * 2 : cur_N + 1000000)) {

      Kokkos::View<int *> x_view("x_view", cur_N);
      Kokkos::View<int *> y_view("y_view", cur_N);

      Kokkos::parallel_for(
          "init", cur_N, KOKKOS_LAMBDA(const int i) {
            y_view(i) = Y_VAL;
            x_view(i) = X_VAL;
          });
      Kokkos::fence();

      double time_scalar_kokkos_avg = run_test_avg(
          [&]() { test_scalar_kokkos(cur_N, A_VAL, x_view, y_view); }, true);

      double time_scalar_base_avg = run_test_avg(
          [&]() { test_scalar_base(cur_N, A_VAL, x_view, y_view); }, true);
      double time_simd_kokkos_avg = run_test_avg(
          [&]() { test_simd_kokkos(cur_N, A_VAL, x_view, y_view); }, true);

      results.push_back({.n = cur_N,
                         .times = {time_scalar_kokkos_avg, time_scalar_base_avg,
                                   time_simd_kokkos_avg}});
    }
  }
  Kokkos::finalize();

  results_to_csv(results);

  // auto n_values = std::map<Result, typename Tp>
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
