#include "bortsova_a_shell_batcher/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stack>
#include <vector>

#include "bortsova_a_shell_batcher/common/include/common.hpp"

namespace bortsova_a_shell_batcher {

namespace {
void SortLocal(std::vector<double> &data) {
  if (data.empty()) {
    return;
  }
  int size = static_cast<int>(data.size());
  std::vector<int> steps = {701, 301, 132, 57, 23, 10, 4, 1};
  for (int step : steps) {
    for (int i = step; i < size; ++i) {
      double value = data[i];
      int pos = i;
      while (pos >= step && data[pos - step] > value) {
        data[pos] = data[pos - step];
        pos -= step;
      }
      data[pos] = value;
    }
  }
}

int FindPower(int n) {
  int result = 1;
  while (result * 2 <= n) {
    result *= 2;
  }
  return result;
}

MPI_Comm CreateComm(MPI_Comm comm, int &new_rank, int &new_size, bool &inside) {
  int old_rank = 0;
  int old_size = 0;
  MPI_Comm_rank(comm, &old_rank);
  MPI_Comm_size(comm, &old_size);
  int power = FindPower(old_size);
  MPI_Group old_group = MPI_GROUP_NULL;
  MPI_Group new_group = MPI_GROUP_NULL;
  MPI_Comm_group(comm, &old_group);
  std::vector<int> ranks(static_cast<size_t>(power));
  for (int i = 0; i < power; ++i) {
    ranks[static_cast<size_t>(i)] = i;
  }
  MPI_Group_incl(old_group, power, ranks.data(), &new_group);
  MPI_Comm new_comm = MPI_COMM_NULL;
  MPI_Comm_create(comm, new_group, &new_comm);
  if (old_rank < power) {
    inside = true;
    MPI_Comm_rank(new_comm, &new_rank);
    MPI_Comm_size(new_comm, &new_size);
  } else {
    inside = false;
    new_rank = -1;
    new_size = 0;
  }
  MPI_Group_free(&old_group);
  if (inside) {
    MPI_Group_free(&new_group);
  }
  return new_comm;
}

void PerfectShuffle(std::vector<double> &data, int start, int end) {
  int size = end - start + 1;
  if (size % 2 != 0) {
    return;
  }
  int half = size / 2;
  std::vector<double> temp(static_cast<size_t>(size));
  for (int i = 0; i < half; ++i) {
    temp[static_cast<int64_t>(2) * i] = data[start + i];
    temp[(2 * i) + 1] = data[start + half + i];
  }
  for (int i = 0; i < size; ++i) {
    data[start + i] = temp[i];
  }
}

void PerfectUnshuffle(std::vector<double> &data, int start, int end) {
  int size = end - start + 1;
  if (size % 2 != 0) {
    return;
  }
  int half = size / 2;
  std::vector<double> temp(static_cast<size_t>(size));
  for (int i = 0; i < half; ++i) {
    temp[i] = data[start + (2 * i)];
    temp[half + i] = data[start + (2 * i) + 1];
  }
  for (int i = 0; i < size; ++i) {
    data[start + i] = temp[i];
  }
}

struct MergeTask {
  int left;
  int right;
  bool need_shuffle;
};

void BatcherOddEvenMerge(std::vector<double> &data, int left, int right) {
  int size = right - left + 1;
  if (size <= 1) {
    return;
  }
  if (size == 2) {
    CompareAndSwap(data[static_cast<size_t>(left)], data[static_cast<size_t>(right)]);
    return;
  }

  std::stack<MergeTask> tasks;
  tasks.push({left, right, true});

  while (!tasks.empty()) {
    MergeTask current = tasks.top();
    tasks.pop();

    int current_left = current.left;
    int current_right = current.right;
    int current_size = current_right - current_left + 1;

    if (current_size <= 2) {
      if (current_size == 2) {
        CompareAndSwap(data[static_cast<size_t>(current_left)], data[static_cast<size_t>(current_right)]);
      }
      continue;
    }

    if (current.need_shuffle) {
      int middle = current_left + (current_size / 2) - 1;

      PerfectUnshuffle(data, current_left, current_right);

      tasks.push({current_left, current_right, false});
      tasks.push({current_left, middle, true});
      tasks.push({middle + 1, current_right, true});
    } else {
      PerfectShuffle(data, current_left, current_right);

      for (int i = current_left; i < current_right; i += 2) {
        CompareAndSwap(data[static_cast<size_t>(i)], data[i + 1]);
      }
    }
  }
}

void MergeSmallPart(std::vector<double> &local, std::vector<double> &other, int size) {
  std::vector<double> merged(static_cast<size_t>(size));
  int i = 0;
  int j = 0;
  int k = 0;
  while (k < size) {
    bool take_local = i < size && (j >= size || local[static_cast<size_t>(i)] < other[static_cast<size_t>(j)]);
    merged[static_cast<size_t>(k)] = take_local ? local[static_cast<size_t>(i)] : other[static_cast<size_t>(j)];
    i += take_local ? 1 : 0;
    j += take_local ? 0 : 1;
    ++k;
  }
  std::ranges::copy(merged, local.begin());
}

void MergeLargePart(std::vector<double> &local, std::vector<double> &other, int size) {
  std::vector<double> merged(static_cast<size_t>(size));
  int i = size - 1;
  int j = size - 1;
  int k = size - 1;
  while (k >= 0) {
    bool take_local = i >= 0 && (j < 0 || local[static_cast<size_t>(i)] > other[static_cast<size_t>(j)]);
    merged[static_cast<size_t>(k)] = take_local ? local[static_cast<size_t>(i)] : other[static_cast<size_t>(j)];
    i -= take_local ? 1 : 0;
    j -= take_local ? 0 : 1;
    --k;
  }
  std::ranges::copy(merged, local.begin());
}

void ParallelBatcherSort(std::vector<double> &data, MPI_Comm comm, int rank, int size) {
  int local_size = static_cast<int>(data.size());
  int max_size = 0;
  MPI_Allreduce(&local_size, &max_size, 1, MPI_INT, MPI_MAX, comm);
  if (local_size < max_size) {
    data.resize(static_cast<size_t>(max_size), std::numeric_limits<double>::max());
  }
  SortLocal(data);
  int cur_size = static_cast<int>(data.size());
  BatcherOddEvenMerge(data, 0, cur_size - 1);
  std::vector<double> buffer(static_cast<size_t>(max_size));
  for (int step = 1; step < size; step *= 2) {
    for (int offset = step; offset > 0; offset /= 2) {
      int partner = rank ^ offset;
      if (partner < size) {
        MPI_Sendrecv(data.data(), max_size, MPI_DOUBLE, partner, 0, buffer.data(), max_size, MPI_DOUBLE, partner, 0,
                     comm, MPI_STATUS_IGNORE);
        if (rank < partner) {
          MergeSmallPart(data, buffer, max_size);
        } else {
          MergeLargePart(data, buffer, max_size);
        }
      }
      MPI_Barrier(comm);
    }
  }
  if (local_size < max_size) {
    data.resize(static_cast<size_t>(local_size));
  }
}

void MergeSegmentPart(std::vector<double> &global, std::vector<double> &temp, int current, int start, int count) {
  std::vector<double> merged(static_cast<size_t>(current + count));
  int i = 0;
  int j = 0;
  int k = 0;
  while (i < current && j < count) {
    bool take_temp = temp[static_cast<size_t>(i)] < global[start + j];
    merged[static_cast<size_t>(k)] = take_temp ? temp[static_cast<size_t>(i)] : global[start + j];
    i += take_temp ? 1 : 0;
    j += take_temp ? 0 : 1;
    ++k;
  }
  while (i < current) {
    merged[static_cast<size_t>(k)] = temp[static_cast<size_t>(i)];
    ++i;
    ++k;
  }
  while (j < count) {
    merged[static_cast<size_t>(k)] = global[start + j];
    ++j;
    ++k;
  }
  for (int idx = 0; idx < current + count; ++idx) {
    temp[static_cast<size_t>(idx)] = merged[static_cast<size_t>(idx)];
  }
}

void FinalMerge(std::vector<double> &global, const std::vector<int> &counts, const std::vector<int> &starts) {
  int total = static_cast<int>(global.size());
  std::vector<double> temp(static_cast<size_t>(total));
  int current = 0;
  int first_count = counts[0];
  for (int i = 0; i < first_count; ++i) {
    temp[static_cast<size_t>(current)] = global[static_cast<size_t>(current)];
    ++current;
  }
  int proc_count = static_cast<int>(counts.size());
  for (int proc = 1; proc < proc_count; ++proc) {
    MergeSegmentPart(global, temp, current, starts[proc], counts[proc]);
    current += counts[proc];
  }
  std::ranges::copy(temp, global.begin());
}

void SortMPI(std::vector<double> &global, MPI_Comm comm) {
  int rank = 0;
  int count = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &count);
  int total = static_cast<int>(global.size());
  int inner_rank = -1;
  int inner_count = 0;
  bool inside = false;
  MPI_Comm inner_comm = CreateComm(comm, inner_rank, inner_count, inside);
  int base = total / count;
  int extra = total % count;
  int local = rank < extra ? base + 1 : base;
  std::vector<int> sizes(static_cast<size_t>(count));
  std::vector<int> offsets(static_cast<size_t>(count));
  int pos = 0;
  for (int i = 0; i < count; ++i) {
    sizes[static_cast<size_t>(i)] = i < extra ? base + 1 : base;
    offsets[static_cast<size_t>(i)] = pos;
    pos += sizes[static_cast<size_t>(i)];
  }
  std::vector<double> part(static_cast<size_t>(local));
  MPI_Scatterv(global.data(), sizes.data(), offsets.data(), MPI_DOUBLE, part.data(), local, MPI_DOUBLE, 0, comm);
  if (inside) {
    ParallelBatcherSort(part, inner_comm, inner_rank, inner_count);
    MPI_Comm_free(&inner_comm);
  } else {
    SortLocal(part);
    int cur = static_cast<int>(part.size());
    BatcherOddEvenMerge(part, 0, cur - 1);
  }
  MPI_Gatherv(part.data(), local, MPI_DOUBLE, global.data(), sizes.data(), offsets.data(), MPI_DOUBLE, 0, comm);
  if (rank == 0) {
    FinalMerge(global, sizes, offsets);
  }
}

}  // namespace

BortsovaAShellBatcherkMPI::BortsovaAShellBatcherkMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool BortsovaAShellBatcherkMPI::ValidationImpl() {
  return true;
}

bool BortsovaAShellBatcherkMPI::PreProcessingImpl() {
  return true;
}

bool BortsovaAShellBatcherkMPI::RunImpl() {
  auto &array = GetInput();
  SortMPI(array, MPI_COMM_WORLD);
  return true;
}

bool BortsovaAShellBatcherkMPI::PostProcessingImpl() {
  return true;
}

}  // namespace bortsova_a_shell_batcher
