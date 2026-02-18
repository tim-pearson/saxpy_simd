#include "kernals.hh"
#include <cassert>

int check_error(Kokkos::View<int *> y, int n) {
  auto y_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  int error = 0;
  int expected = Y_VAL + A_VAL * X_VAL;
  for (int i = 0; i < n; i++)
    error += std::abs(expected - y_host(i));

  assert(error == 0);
  return error;
}

void test_scalar_base(int N, int_1d_view x_view, int_1d_view y_view) {
  for (int i = 0; i < N; i++) {
    y_view[i] = A_VAL * x_view[i] + y_view[i];
  }
}

void test_simd_kokkos(int N, int_1d_view x_view, int_1d_view y_view) {
  const int simd_size = simd_t::size();
  int stride = (N + simd_size - 1) / simd_size;

  Kokkos::parallel_for(
      "simd_saxpy", stride, KOKKOS_LAMBDA(const int i) {
        simd_t x_simd(&x_view(i * simd_size), KE::simd_flag_default);
        simd_t y_simd(&y_view(i * simd_size), KE::simd_flag_default);
        y_simd = y_simd + x_simd * A_VAL;
        KE::simd_unchecked_store(y_simd, &y_view(i * simd_size),
                                 KE::simd_flag_default);
      });
}

void test_scalar_kokkos(int N, int_1d_view x_view, int_1d_view y_view) {
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::Serial>(0, N), KOKKOS_LAMBDA(const int i) {
        y_view[i] = A_VAL * x_view[i] + y_view[i];
      });
}
