#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_SIMD.hpp>
#include <impl/Kokkos_InitializeFinalize.hpp>
#include <matplot/freestanding/plot.h>
#include <matplot/matplot.h>

namespace KE = Kokkos::Experimental;
using simd_t = KE::simd<int>;
using int_1d_view = Kokkos::View<int *>;
int check_error(Kokkos::View<int *> y, int n);

void test_scalar_base(int N, int a, int_1d_view x_view, int_1d_view y_view);

void test_simd_kokkos(int N_in, int a, int_1d_view x_view, int_1d_view y_view);

void test_scalar_kokkos(int N, int a, int_1d_view x_view, int_1d_view y_view);
