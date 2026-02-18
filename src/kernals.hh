#include "consts.hh"
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_SIMD.hpp>
#include <impl/Kokkos_InitializeFinalize.hpp>
#include <matplot/freestanding/plot.h>
#include <matplot/matplot.h>

// constexpr int N_MAX = 1000000;
// constexpr int A_VAL = 3;
// constexpr int X_VAL = 3;
// constexpr int Y_VAL = 2;
// constexpr int REPEAT_COUNT = 5;

namespace KE = Kokkos::Experimental;
using simd_t = KE::simd<int>;
using int_1d_view = Kokkos::View<int *>;
int check_error(Kokkos::View<int *> y, int n);

void test_scalar_base(int N, int_1d_view x_view, int_1d_view y_view);

void test_simd_kokkos(int N, int_1d_view x_view, int_1d_view y_view);

void test_scalar_kokkos(int N, int_1d_view x_view, int_1d_view y_view);
