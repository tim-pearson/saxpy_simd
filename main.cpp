#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_SIMD.hpp>
#include <impl/Kokkos_InitializeFinalize.hpp>
#include <iostream>
using simd_t = Kokkos::Experimental::simd<int>;

void print_error(Kokkos::View<int *> y, int n) {
  int error = 0.0;

  for (int i = 0; i < n; i++) {
    error += std::abs(11 - y(i));
    // std::cout << "y = " << y(i) << '\n';
  }

  std::cout << "Error: " << error << std::endl;
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
  std::cout << "simd time = " << time << '\n';
  print_error(y, N_in);
  return time;
}

double test_scalar(int N, int a, int x_val, int y_val) {
  const int simd_size = simd_t::size();

  Kokkos::View<int *> x("x", N * simd_size);
  Kokkos::View<int *> y("y", N * simd_size);
  Kokkos::parallel_for(
      "init", N * simd_size, KOKKOS_LAMBDA(const int i) {
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
  std::cout << "scalar time = " << time << '\n';
  print_error(y, N);
  return time;
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    Kokkos::print_configuration(std::cout);

    int N = 10000000;
    // int N = 1000;
    int a = 3;
    std::cout << "N = " << N << '\n';
    std::cout << "a = " << a << '\n';
    std::cout << "simd size: " << simd_t::size() << '\n';
    std::cout << "N /simd size: " << N / simd_t::size() << '\n';

    int x_val = 3;
    int y_val = 2;
    double scalar_time = test_scalar(N, a, x_val, y_val);
    double simd_time = test_simd(N, a, x_val, y_val);
    std::cout << "speedup = " << scalar_time / simd_time << '\n';
  }
  Kokkos::finalize();
}
