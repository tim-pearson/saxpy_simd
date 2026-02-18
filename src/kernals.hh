#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_SIMD.hpp>
#include <impl/Kokkos_InitializeFinalize.hpp>
#include <matplot/freestanding/plot.h>
#include <matplot/matplot.h>

using simd_t = Kokkos::Experimental::simd<int>;

int check_error(Kokkos::View<int *> y, int n);

double test_scalar_base(int N, int a, int x_val, int y_val);

double test_simd_kokkos(int N_in, int a, int x_val, int y_val);

double test_scalar_kokkos(int N, int a, int x_val, int y_val);
