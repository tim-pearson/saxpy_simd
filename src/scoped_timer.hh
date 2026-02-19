#include <chrono>

#include <Kokkos_Core.hpp>
using std_clock = std::chrono::steady_clock;

class ScopedTimer {
public:
  ScopedTimer(double &out, bool isKokkos = false);
  ~ScopedTimer();

private:
  bool do_fence;
  double &output;
  std_clock::time_point start;
};
