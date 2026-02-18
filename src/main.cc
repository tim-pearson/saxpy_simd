#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_SIMD.hpp>
#include <chrono>
#include <impl/Kokkos_InitializeFinalize.hpp>
#include <iostream>
#include <matplot/freestanding/plot.h>
#include <matplot/matplot.h>
#include <vector>

using simd_t = Kokkos::Experimental::simd<int>;
using std_clock = std::chrono::steady_clock;

class ScopedTimer {
public:
  ScopedTimer(double &out, bool isKokkos = false)
      : do_fence(isKokkos), output(out), start(std_clock::now()) {}
  ~ScopedTimer() {
    if (do_fence)
      Kokkos::fence();
    auto end = std_clock::now();
    output = std::chrono::duration<double>(end - start).count();
  }

private:
  bool do_fence;
  double &output;
  std_clock::time_point start;
};

int check_error(Kokkos::View<int *> y, int n) {
  int error = 0.0;

  for (int i = 0; i < n; i++) {
    error += std::abs(11 - y(i));
    // std::cout << "y = " << y(i) << '\n';
  }

  if (error > 0.000001)
    std::cout << "============ERROR: " << error << std::endl;
  return error;
}

double test_scalar_base(int N, int a, int x_val, int y_val) {
  const int simd_size = simd_t::size();

  std::vector<int> x(x_val, N);
  std::vector<int> y(y_val, N);
  for (int i = 0; i < N; i++) {
    y[i] = a * x[i] + y[i];
  }
  Kokkos::Timer timer;
  timer.reset();

  double time = timer.seconds();
  return time;
}

double test_simd_kokkos(int N_in, int a, int x_val, int y_val) {
  const int simd_size = simd_t::size();
  int N = (N_in + simd_size - 1) / simd_size;

  Kokkos::View<int *> x("x", N * simd_size);
  Kokkos::View<int *> y("y", N * simd_size);
  Kokkos::parallel_for(
      "init", N * simd_size, KOKKOS_LAMBDA(const int i) {
        y(i) = y_val;
        x(i) = x_val;
      });

  Kokkos::fence();

  Kokkos::Timer timer;
  Kokkos::parallel_for(
      "simd_saxpy", N, KOKKOS_LAMBDA(const int i) {
        simd_t x_(&x(i * simd_size), Kokkos::Experimental::simd_flag_default);
        simd_t y_(&y(i * simd_size), Kokkos::Experimental::simd_flag_default);
        y_ = y_ + x_ * a;
        Kokkos::Experimental::simd_unchecked_store(
            y_, &y(i * simd_size), Kokkos::Experimental::simd_flag_default);
      });
  Kokkos::fence();
  double time = timer.seconds();
  int error = check_error(y, N_in);

  return time;
}

double test_scalar_kokkos(int N, int a, int x_val, int y_val) {
  const int simd_size = simd_t::size();

  Kokkos::View<int *> x("x", N * simd_size);
  Kokkos::View<int *> y("y", N * simd_size);
  Kokkos::parallel_for(
      "init", N, KOKKOS_LAMBDA(const int i) {
        y(i) = y_val;
        x(i) = x_val;
      });

  Kokkos::fence();

  Kokkos::Timer timer;
  timer.reset();
  Kokkos::parallel_for(
      "SAXPY scalar", N,
      KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
  Kokkos::fence();

  double time = timer.seconds();
  int error = check_error(y, N);
  return time;
}

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

      Kokkos::View<int *> x("x_view", cur_N);
      Kokkos::View<int *> y("y_view", cur_N);

      Kokkos::parallel_for(
          "init", cur_N, KOKKOS_LAMBDA(const int i) {
            y(i) = y_val;
            x(i) = x_val;
          });
      Kokkos::fence();

      double time_scalar_kokkos;
      {
        ScopedTimer timer(time_scalar_kokkos, true);
        test_scalar_kokkos(cur_N, a_val, x_val, y_val);
      }
      double time_simd_kokkos;
      {
        ScopedTimer timer(time_simd_kokkos, true);
        test_scalar_kokkos(cur_N, a_val, x_val, y_val);
      }
      double time_scalar_base;
      {
        ScopedTimer timer(time_scalar_base, false);
        test_scalar_kokkos(cur_N, a_val, x_val, y_val);
      }
    }
  }
  Kokkos::finalize();

  std::ofstream outFile("benchmark_results.csv");
  outFile << "N,scalar_base,scalar_kokkos,simd_kokkos,Speedup\n";
  for (size_t i = 0; i < n_values.size(); ++i) {
    outFile << n_values[i] << "," << times_scalar[i] << "," << times_simd[i]
            << "," << speedups[i] << "\n";
  }
  outFile.close();

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
