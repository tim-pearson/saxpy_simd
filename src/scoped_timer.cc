#include "scoped_timer.hh"

ScopedTimer::ScopedTimer(double &out, bool isKokkos)
    : do_fence(isKokkos), output(out), start(std_clock::now()) {}

ScopedTimer::~ScopedTimer() {
  if (do_fence)
    Kokkos::fence();
  auto end = std_clock::now();
  output = std::chrono::duration<double>(end - start).count();
}
