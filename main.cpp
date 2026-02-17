#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_SIMD.hpp>
#include <impl/Kokkos_InitializeFinalize.hpp>
#include <iostream>
#include <matplot/freestanding/plot.h>
#include <matplot/matplot.h>
#include <vector>
using simd_t = Kokkos::Experimental::simd<int>;

void check_error(Kokkos::View<int *> y, int n) {
  int error = 0.0;

  for (int i = 0; i < n; i++) {
    error += std::abs(11 - y(i));
    // std::cout << "y = " << y(i) << '\n';
  }

  if (error > 0.000001)
    std::cout << "============ERROR: " << error << std::endl;
}
double test_simd(int N_in, int a, int x_val, int y_val) {
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
  // std::cout << "simd time = " << time << '\n';
  check_error(y, N_in);
  return time;
}

double test_scalar(int N, int a, int x_val, int y_val) {
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
  // std::cout << "scalar time = " << time << '\n';
  check_error(y, N);
  return time;
}

// int main(int argc, char *argv[]) {
//   std::vector<double> times_simd;
//   std::vector<double> times_scalar;
//   std::vector<int> n_values;
//   Kokkos::initialize(argc, argv);
//   {

//     int n_max = 10000000;
//     int a = 3;

//     for (int i = 200; i <= n_max; i = (i < 100000 ? i * 2 : i + 1000000)) {
//       std::cout << "N = " << i << '\n';
//       std::cout << "a = " << a << '\n';
//       std::cout << "SIMD loops iterations " << i / simd_t::size() << '\n';

//       int x_val = 3;
//       int y_val = 2;
//       double scalar_time = 0;
//       double simd_time = 0;
//       for (int j = 0; j < 5; j++) {
//         scalar_time += test_scalar(i, a, x_val, y_val);
//         simd_time += test_simd(i, a, x_val, y_val);
//       }
//       double avg_scalar = scalar_time / 5;
//       double avg_simd = simd_time / 5;
//       times_scalar.push_back(avg_scalar);
//       times_simd.push_back(avg_simd);
//       n_values.push_back(i);

//       std::cout << "speedup = " << scalar_time / simd_time << "\n\n";
//     }
//   }
//   Kokkos::finalize();
//   matplot::plot(n_values, times_simd);
//   matplot::show();
// }

int main(int argc, char *argv[]) {
  std::vector<double> times_simd;
  std::vector<double> times_scalar;
  std::vector<int> n_values;
  std::vector<double> speedups;
  Kokkos::initialize(argc, argv);
  {
    int n_max = 10000000;
    int a = 3;

    for (int i = 200; i <= n_max; i = (i < 100000 ? i * 2 : i + 1000000)) {
      int x_val = 3;
      int y_val = 2;
      double scalar_time = 0;
      double simd_time = 0;
      for (int j = 0; j < 5; j++) {
        scalar_time += test_scalar(i, a, x_val, y_val);
        simd_time += test_simd(i, a, x_val, y_val);
      }
      double avg_scalar = scalar_time / 5;
      double avg_simd = simd_time / 5;
      times_scalar.push_back(avg_scalar);
      times_simd.push_back(avg_simd);
      n_values.push_back(i);
      speedups.push_back(avg_scalar / avg_simd);
    }
  }
  Kokkos::finalize();

  auto fig1 = matplot::figure();
  matplot::semilogx(n_values, times_scalar)
      ->line_width(2)
      .display_name("Scalar");
  matplot::hold(matplot::on);
  matplot::semilogx(n_values, times_simd)->line_width(2).display_name("SIMD");
  matplot::xlabel("N");
  matplot::ylabel("Time (s)");
  matplot::title("Scalar vs SIMD Runtime");
  matplot::legend();
  matplot::grid(matplot::on);

  auto fig2 = matplot::figure();
  matplot::semilogx(n_values, speedups)->line_width(2).display_name("Speedup");
  matplot::xlabel("N");
  matplot::ylabel("Speedup");
  matplot::title("SIMD Speedup vs Scalar");
  matplot::grid(matplot::on);

  matplot::show();
}
