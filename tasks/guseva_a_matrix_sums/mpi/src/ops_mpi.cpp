#include "guseva_a_matrix_sums/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstdint>

namespace guseva_a_matrix_sums {

GusevaAMatrixSumsMPI::GusevaAMatrixSumsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool GusevaAMatrixSumsMPI::ValidationImpl() {
  return (static_cast<uint64_t>(std::get<0>(GetInput())) * std::get<1>(GetInput()) == std::get<2>(GetInput()).size()) &&
         (GetOutput().empty());
}

bool GusevaAMatrixSumsMPI::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().resize(std::get<1>(GetInput()), 0.0);
  return true;
}

bool GusevaAMatrixSumsMPI::RunImpl() {
  int rows = 0;
  int columns = 0;
  std::vector<double> matrix;
  int wsize = 0;
  int rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &wsize);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    rows = static_cast<int>(std::get<0>(GetInput()));
    columns = static_cast<int>(std::get<1>(GetInput()));
    matrix = std::get<2>(GetInput());
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);

  uint32_t rows_per_proc = rows / wsize;
  uint32_t remainder = rows % wsize;

  std::vector<int> displs(wsize, 0);
  std::vector<int> counts;
  counts.reserve(wsize);

  if (rank == 0) {
    for (int rnk = 0; rnk < wsize; rnk++) {
      uint32_t start_row = (rnk * rows_per_proc) + std::min(static_cast<uint32_t>(rnk), remainder);
      uint32_t end_row = ((rnk + 1) * rows_per_proc) + std::min(static_cast<uint32_t>(rnk + 1), remainder);
      uint32_t start_pos = start_row * columns;
      uint32_t end_pos = end_row * columns;
      counts.push_back(static_cast<int>(end_pos - start_pos));
      for (int i = rnk + 1; i < wsize; i++) {
        displs[i] += counts.back();
      }
    }
  }

  uint32_t start_row = (rank * rows_per_proc) + std::min(static_cast<uint32_t>(rank), remainder);
  uint32_t end_row = ((rank + 1) * rows_per_proc) + std::min(static_cast<uint32_t>(rank + 1), remainder);

  uint32_t start_pos = start_row * columns;
  uint32_t end_pos = end_row * columns;

  std::vector<double> slice(end_pos - start_pos, 0);
  MPI_Scatterv(matrix.data(), counts.data(), displs.data(), MPI_DOUBLE, slice.data(),
               static_cast<int>(end_pos - start_pos), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  std::vector<double> local_sums(columns, 0);
  for (uint32_t i = 0; i < end_pos - start_pos; i++) {
    local_sums[i % columns] += slice[i];
  }

  std::vector<double> global_sums(columns, 0);
  MPI_Reduce(local_sums.data(), global_sums.data(), columns, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(global_sums.data(), columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput().assign(global_sums.begin(), global_sums.end());
  return true;
}

bool GusevaAMatrixSumsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace guseva_a_matrix_sums
