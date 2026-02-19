#include "consts.hh"
#include "scoped_timer.hh"
#include <cmath>
#include <fstream>
#include <matplot/freestanding/plot.h>
#include <matplot/matplot.h>
#include <vector>

using namespace matplot;
struct Result {
  int n;
  double times[3]; // store one time for each testcase (0:scalar_kokkos,
                   // 1:scalar_base, 2:simd_kokkos)
};

void results_to_csv(const std::vector<Result> &results);

/**
 * runs a REPEAT_COUNT repetions for a single test case
 *
 * @param test_func: function from kernals.hh with agrs already passed
 * @param reset: resets the x_view and y_view with inital values (for error
 * @param error_check: checks if the expected matches the y_view values
 * @param is_kokkos: passed to ScopedTimer, if true then called Kokkos::fence()
 * @return average run time of kernal for a given vector length
 */
template <typename TestFunc, typename ResetFunc, typename ErrorFunc>
double run_test_avg(TestFunc &&test_func, ResetFunc &&reset,
                    ErrorFunc error_check, bool is_kokkos) {
  //
  double total_time = 0.0;

  for (int i = 0; i < REPEAT_COUNT; i++) {

    reset();
    double run_time = 0.0;
    {
      ScopedTimer timer(run_time, is_kokkos);
      test_func();
    }
    error_check();
    total_time += run_time;
  }

  return total_time / REPEAT_COUNT;
}

void plot_results(const std::vector<Result> &results);
