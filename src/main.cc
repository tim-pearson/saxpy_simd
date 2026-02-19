#include "consts.hh"
#include "kernals.hh"
#include "utils.hh"
#include <Kokkos_Core_fwd.hpp>
#include <cassert>
#include <impl/Kokkos_InitializeFinalize.hpp>

int main(int argc, char *argv[]) {

  std::vector<Result> results;
  Kokkos::initialize(argc, argv);
  {

    for (int cur_N = 1000; cur_N <= N_MAX; cur_N *= 2) {

      Kokkos::View<int *> x_view("x_view", cur_N);
      Kokkos::View<int *> y_view("y_view", cur_N);

      const int simd_size = simd_t::size();
      assert(cur_N % simd_size == 0);

      auto reset = [&]() {
        Kokkos::parallel_for(
            "reset", cur_N, KOKKOS_LAMBDA(const int i) {
              y_view(i) = Y_VAL;
              x_view(i) = X_VAL;
            });
        Kokkos::fence();
      };
      auto error_check = [&]() { check_error(y_view, cur_N); };

      Kokkos::fence();

      double scalar_k_time =
          run_test_avg([&]() { test_scalar_kokkos(cur_N, x_view, y_view); },
                       reset, error_check, true);

      double simd_k_time =
          run_test_avg([&]() { test_simd_kokkos(cur_N, x_view, y_view); },
                       reset, error_check, true);

      double scalar_b_time =
          run_test_avg([&]() { test_scalar_base(cur_N, x_view, y_view); },
                       reset, error_check, false);

      results.push_back(
          {.n = cur_N, .times = {scalar_k_time, scalar_b_time, simd_k_time}});
    }
  }
  Kokkos::finalize();

  results_to_csv(results);
  plot_results(results);
  return 0;
}
