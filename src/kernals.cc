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

void test_scalar_base(int N, int a, int_1d_view x_view, int_1d_view y_view) {
  const int simd_size = simd_t::size();

  for (int i = 0; i < N; i++) {
    y_view[i] = a * x_view[i] + y_view[i];
  }
}

void test_simd_kokkos(int N_in, int a, int_1d_view x_view, int_1d_view y_view) {
  const int simd_size = simd_t::size();
  int N = (N_in + simd_size - 1) / simd_size;

  Kokkos::parallel_for(
      "simd_saxpy", N, KOKKOS_LAMBDA(const int i) {
        simd_t x_simd(&x_view(i * simd_size), KE::simd_flag_default);
        simd_t y_simd(&y_view(i * simd_size), KE::simd_flag_default);
        y_simd = y_simd + x_simd * a;
        KE::simd_unchecked_store(y_simd, &y_view(i * simd_size),
                                 KE::simd_flag_default);
      });
}

void test_scalar_kokkos(int N, int a, int_1d_view x_view, int_1d_view y_view) {
  const int simd_size = simd_t::size();

  // Kokkos::parallel_for(
  //     "SAXPY scalar", N,
  //     KOKKOS_LAMBDA(const int i) { y_view[i] = a * x_view[i] + y_view[i]; });

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::Serial>(0, N),
      KOKKOS_LAMBDA(const int i) { y_view[i] = a * x_view[i] + y_view[i]; });

  // Kokkos::single_task(KOKKOS_LAMBDA() {
  //   for (int i = 0; i < N; i++) {
  //     y_view[i] = a * x_view[i] + y_view[i];
  //   }
  // });
}
