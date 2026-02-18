#include "utils.hh"
#include "gnuplot.h"
void results_to_csv(const std::vector<Result> &results) {
  // save results to csv file
  // cols = |len|scalar_kokkos|-speedup->|scalar_base|-speedup->|//simd_kokkos|
  std::ofstream outFile("benchmark_results.csv");

  outFile << "N,scalar_kokkos,--->,scalar_base,--->,simd_kokkos\n";

  for (const auto &res : results) {
    double speedup_1 =
        std::round((res.times[0] / res.times[1]) * 1000.0) / 1000.0;
    double speedup_2 =
        std::round((res.times[1] / res.times[2]) * 1000.0) / 1000.0;

    outFile << res.n << "," << res.times[0] << ",x" << speedup_1 << ","
            << res.times[1] << ",x" << speedup_2 << "," << res.times[2] << "\n";
  }

  outFile.close();
}

void plot_results(const std::vector<Result> &results) {
  GnuplotPipe gp;

  // Set up log-log scale, labels, title, grid
  gp.sendLine("set logscale xy");
  gp.sendLine("set xlabel 'Problem size N'");
  gp.sendLine("set ylabel 'Time (seconds)'");
  gp.sendLine("set title 'SAXPY Benchmark Results'");
  gp.sendLine("set grid");

  // Start plot command with three inline datasets
  gp.sendLine(
      "plot '-' with linespoints lt 1 lc rgb 'red' pt 7 title 'Scalar Kokkos', "
      "'-' with linespoints lt 1 lc rgb 'green' pt 5 title 'Scalar Base', "
      "'-' with linespoints lt 1 lc rgb 'blue' pt 9 title 'SIMD Kokkos'");

  // Send Scalar Kokkos data
  for (const auto &r : results)
    gp.sendLine(std::to_string(r.n) + " " + std::to_string(r.times[0]));
  gp.sendEndOfData();

  // Send Scalar Base data
  for (const auto &r : results)
    gp.sendLine(std::to_string(r.n) + " " + std::to_string(r.times[1]));
  gp.sendEndOfData();

  // Send SIMD Kokkos data
  for (const auto &r : results)
    gp.sendLine(std::to_string(r.n) + " " + std::to_string(r.times[2]));
  gp.sendEndOfData();
}
