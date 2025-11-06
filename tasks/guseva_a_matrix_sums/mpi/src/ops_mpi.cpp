#include "guseva_a_matrix_sums/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstdint>
#include <vector>

#include "guseva_a_matrix_sums/common/include/common.hpp"

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
  return true;
}

bool GusevaAMatrixSumsMPI::RunImpl() {
  int rank = 0;
  int wsize = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &wsize);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int rows = static_cast<int>(std::get<0>(GetInput()));
  int columns = static_cast<int>(std::get<1>(GetInput()));

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<double> matrix(static_cast<int64_t>(rows) * columns, 0.0);
  if (rank == 0) {
    matrix = std::get<2>(GetInput());
  }

  int rows_per_proc = rows / wsize;
  int remainder = rows % wsize;
  std::vector<int> counts(wsize, 0);
  std::vector<int> displs(wsize, 0);
  int displ = 0;
  for (int proc = 0; proc < wsize; proc++) {
    counts[proc] = (rows_per_proc + (proc < remainder ? 1 : 0)) * columns;
    displs[proc] = displ;
    displ += counts[proc];
  }
  int local_count = counts[rank];
  std::vector<double> local_matrix(local_count, 0);

  MPI_Scatterv(matrix.data(), counts.data(), displs.data(), MPI_DOUBLE, local_matrix.data(), local_count, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  std::vector<double> local_sums(columns, 0.0);
  for (int i = 0; i < local_count; i++) {
    local_sums[i % columns] += local_matrix[i];
  }
  std::vector<double> global_sums(columns, 0.0);
  MPI_Reduce(local_sums.data(), global_sums.data(), columns, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  GetOutput() = {-1};
  if (rank == 0) {
    GetOutput() = global_sums;
  }
  return true;
}

bool GusevaAMatrixSumsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace guseva_a_matrix_sums
