#include "kernals.hh"
#include "utils.hh"
#include <Kokkos_Core_fwd.hpp>
#include <functional>
#include <impl/Kokkos_InitializeFinalize.hpp>
#include <matplot/freestanding/plot.h>
#include <matplot/matplot.h>

int main(int argc, char *argv[]) {

  std::vector<Result> results;
  Kokkos::initialize(argc, argv);
  {

    for (int cur_N = 9000; cur_N <= N_MAX;
         cur_N = (cur_N < 10000 ? cur_N * 2 : cur_N + 100000)) {

      Kokkos::View<int *> x_view("x_view", cur_N);
      Kokkos::View<int *> y_view("y_view", cur_N);

      auto reset = [&]() {
        Kokkos::parallel_for(
            "reset", cur_N, KOKKOS_LAMBDA(const int i) {
              y_view(i) = Y_VAL;
              x_view(i) = X_VAL;
            });
      };

      Kokkos::fence();

      double scalar_k_time = run_test_avg(
          [&]() { test_scalar_kokkos(cur_N, x_view, y_view); }, reset, true);
      check_error(y_view, cur_N);

      double simd_k_time = run_test_avg(
          [&]() { test_simd_kokkos(cur_N, x_view, y_view); }, reset, true);
      check_error(y_view, cur_N);

      double scalar_b_time = run_test_avg(
          [&]() { test_scalar_base(cur_N, x_view, y_view); }, reset, false);
      check_error(y_view, cur_N);

      results.push_back(
          {.n = cur_N, .times = {scalar_k_time, scalar_b_time, simd_k_time}});
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
