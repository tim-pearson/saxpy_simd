#include "kernals.hh"
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
