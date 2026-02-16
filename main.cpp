#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_SIMD.hpp>
#include <impl/Kokkos_InitializeFinalize.hpp>
#include <iostream>
using simd_t = Kokkos::Experimental::simd<double>;

void print_error(Kokkos::View<double *> y, int n) {
  int error = 0.0;

  for (int i = 0; i < n; i++) {
    error += std::abs(11 - y(i));
    std::cout << "y = " << y(i) << '\n';
  }

  std::cout << "Error: " << error << std::endl;
}
void test_simd(int N_in, int a) {
  const int simd_size = simd_t::size();
  int N = (N_in + simd_size - 1) / simd_size;

  Kokkos::View<double *> x("x", N * simd_size);
  Kokkos::View<double *> y("y", N * simd_size);

  Kokkos::parallel_for(
      "init", N * simd_size, KOKKOS_LAMBDA(const int i) {
        y(i) = 2;
        x(i) = 3;
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
}

void test_scalar(int N, int a) {

  Kokkos::View<double *> y("y data", N);
  Kokkos::View<double *> x("x data", N);

  Kokkos::parallel_for("init y", N, KOKKOS_LAMBDA(const int i) { y(i) = 2; });
  Kokkos::parallel_for("init x", N, KOKKOS_LAMBDA(const int i) { x(i) = 3; });

  Kokkos::Timer timer;
  timer.reset();
  Kokkos::parallel_for(
      "SAXPY scalar", N,
      KOKKOS_LAMBDA(const int i) { y[i] = a * x[i] + y[i]; });
  Kokkos::fence();

  double time = timer.seconds();
  std::cout << "scalar time = " << time << '\n';
  print_error(y, N);
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout);

  int N = 8;
  int a = 3;
  std::cout << "N = " << N << '\n';
  std::cout << "a = " << a << '\n';
  std::cout << "simd size: " << simd_t::size() << '\n';

  test_scalar(N, a);
  test_simd(N, a);

  Kokkos::finalize();
}
