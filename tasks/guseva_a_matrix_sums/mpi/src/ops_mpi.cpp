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

  int rows_per_proc = rows / wsize;
  int remainder = rows % wsize;

  std::vector<int> displs(wsize, 0);
  std::vector<int> counts(wsize, 0);

  if (rank == 0) {
    int current_displacement = 0;
    for (int proc = 0; proc < wsize; proc++) {
      int proc_rows = rows_per_proc + (proc < remainder ? 1 : 0);

      counts[proc] = proc_rows * columns;
      displs[proc] = current_displacement;

      current_displacement += counts[proc];
      // int start_row = (proc * rows_per_proc) + std::min(proc, remainder);
      // int end_row = ((proc + 1) * rows_per_proc) + std::min(proc, remainder);
      // int row_num = end_row - start_row;
      // counts[proc] = row_num * columns;
      // displs[proc] = displ;
      // displ += counts[proc];
    }
  }

  int local_count = 0;
  int local_displ = 0;

  MPI_Scatter(counts.data(), 1, MPI_INT, &local_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Scatter(displs.data(), 1, MPI_INT, &local_displ, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<double> local_slice(local_count, 0);
  MPI_Scatterv(matrix.data(), counts.data(), displs.data(), MPI_DOUBLE, local_slice.data(), local_count, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  std::vector<double> local_sums(columns, 0);
  for (int i = 0; i < local_count; i++) {
    local_sums[i % columns] += local_slice[i];
  }

  std::vector<double> global_sums(columns, 0);
  MPI_Reduce(local_sums.data(), global_sums.data(), columns, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    GetOutput().assign(global_sums.begin(), global_sums.end());
  } else {
    GetOutput() = {-1};
  }
  return true;
}

bool GusevaAMatrixSumsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace guseva_a_matrix_sums
