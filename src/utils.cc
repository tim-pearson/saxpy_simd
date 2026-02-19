#include "utils.hh"
#include "gnuplot.h"
#include <string>

void results_to_csv(const std::vector<Result> &results) {
  std::string filename = std::string(BUILD_NAME) + "_results.csv";
  std::ofstream outFile(filename);

  outFile << "N,Vector Size (Bytes),Cache "
             "Level,scalar_kokkos,--->,scalar_base,--->,simd_kokkos\n";

  for (const auto &res : results) {
    size_t vector_size_bytes = res.n * sizeof(int);
    std::string cache_level;

    if (vector_size_bytes < 64 * 1024) { // L1: < 64 KB
      cache_level = "L1";
    } else if (vector_size_bytes < 2 * 1024 * 1024) { // L2: < 2 MB
      cache_level = "L2";
    } else if (vector_size_bytes < 12 * 1024 * 1024) { // L3: < 12 MB
      cache_level = "L3";
    } else { // RAM: >= 12 MB
      cache_level = "RAM";
    }

    double speedup_1 =
        std::round((res.times[0] / res.times[1]) * 1000.0) / 1000.0;
    double speedup_2 =
        std::round((res.times[1] / res.times[2]) * 1000.0) / 1000.0;

    outFile << res.n << "," << vector_size_bytes << "," << cache_level << ","
            << res.times[0] << ",x" << speedup_1 << "," << res.times[1] << ",x"
            << speedup_2 << "," << res.times[2] << "\n";
  }

  outFile.close();
}

void plot_results(const std::vector<Result> &results) {
  GnuplotPipe gp;
  std::string filename = std::string(BUILD_NAME) + "_plot.png";
  gp.sendLine("set terminal pngcairo size 1200,800 enhanced font 'Arial,14'");
  gp.sendLine("set output '" + filename + "'");

  // gp.sendLine("set logscale y");
  gp.sendLine("set xrange [" + std::to_string(results.front().n) + ":]");

  gp.sendLine("set xlabel 'Problem size N'");
  gp.sendLine("set ylabel 'Time (seconds)'");
  gp.sendLine("set title 'SAXPY Benchmark " + std::string(BUILD_NAME) + "'");
  gp.sendLine("set grid");

  gp.sendLine(
      "plot '-' with linespoints lt 1 lc rgb 'red' pt 7 title 'Scalar Kokkos', "
      "'-' with linespoints lt 1 lc rgb 'green' pt 5 title 'Scalar Base', "
      "'-' with linespoints lt 1 lc rgb 'blue' pt 9 title 'SIMD Kokkos'");

  for (const auto &r : results)
    gp.sendLine(std::to_string(r.n) + " " + std::to_string(r.times[0]));
  gp.sendEndOfData();

  for (const auto &r : results)
    gp.sendLine(std::to_string(r.n) + " " + std::to_string(r.times[1]));
  gp.sendEndOfData();

  for (const auto &r : results)
    gp.sendLine(std::to_string(r.n) + " " + std::to_string(r.times[2]));
  gp.sendEndOfData();
}
