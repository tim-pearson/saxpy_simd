#include "kernals.hh"

int check_error(Kokkos::View<int *> y, int n) {

  auto y_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);

  int error = 0;

  for (int i = 0; i < n; i++) {
    error += std::abs(11 - y_host(i));
  }

  if (error != 0)
    std::cout << "============ ERROR: " << error << std::endl;

  return error;
}

void test_scalar_base(int N, int a, int_1d_view x_view, int_1d_view y_view) {
  for (int i = 0; i < N; i++) {
    y_view[i] = a * x_view[i] + y_view[i];
  }
}

void test_simd_kokkos(int N, int a, int_1d_view x_view, int_1d_view y_view) {
  const int simd_size = simd_t::size();
  int stride = (N + simd_size - 1) / simd_size;

  Kokkos::parallel_for(
      "simd_saxpy", stride, KOKKOS_LAMBDA(const int i) {
        simd_t x_simd(&x_view(i * simd_size), KE::simd_flag_default);
        simd_t y_simd(&y_view(i * simd_size), KE::simd_flag_default);
        y_simd = y_simd + x_simd * a;
        KE::simd_unchecked_store(y_simd, &y_view(i * simd_size),
                                 KE::simd_flag_default);
      });
}

void test_scalar_kokkos(int N, int a, int_1d_view x_view, int_1d_view y_view) {
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::Serial>(0, N),
      KOKKOS_LAMBDA(const int i) { y_view[i] = a * x_view[i] + y_view[i]; });
}
