#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "task/include/task.hpp"

namespace bortsova_a_shell_batcher {

using InType = std::vector<double>;
using OutType = std::vector<double>;
using TestType = int;
using BaseTask = ppc::task::Task<InType, OutType>;

inline void CompareAndSwap(double &a, double &b) {
  if (a > b) {
    std::swap(a, b);
  }
}

inline void CompareWithinBlock(std::vector<double> &array, int right, int block_start, int block_size) {
  for (int i = 0; i < block_size; ++i) {
    int first_idx = block_start + i;
    int second_idx = block_start + i + block_size;

    if (second_idx <= right) {
      CompareAndSwap(array[first_idx], array[second_idx]);
    }
  }
}

inline void OddEvenMergeBetweenBlocks(std::vector<double> &array, int left, int right, int block_size) {
  for (int block_start = left + block_size; block_start <= right; block_start += 2 * block_size) {
    CompareWithinBlock(array, right, block_start, block_size);
  }
}

inline void FinalNeighborCheck(std::vector<double> &array, int left, int right) {
  for (int i = left; i < right; ++i) {
    CompareAndSwap(array[i], array[i + 1]);
  }
}

inline void BatcherMergeSimpleIterative(std::vector<double> &array, int left, int right) {
  int total_size = right - left + 1;

  if (total_size <= 1) {
    return;
  }

  if (total_size == 2) {
    CompareAndSwap(array[left], array[right]);
    return;
  }

  int block_size = 1;

  while (block_size < total_size) {
    for (int block_start = left; block_start <= right; block_start += 2 * block_size) {
      CompareWithinBlock(array, right, block_start, block_size);
    }
    OddEvenMergeBetweenBlocks(array, left, right, block_size);
    block_size *= 2;
  }

  FinalNeighborCheck(array, left, right);
}

inline void ShellSortBatcherMerge(std::vector<double> &array) {
  if (array.empty()) {
    return;
  }

  int array_size = static_cast<int>(array.size());

  std::vector<int> gaps = {701, 301, 132, 57, 23, 10, 4, 1};

  for (int current_gap : gaps) {
    for (int i = current_gap; i < array_size; ++i) {
      double temp_element = array[i];
      int j = i;
      while (j >= current_gap && array[j - current_gap] > temp_element) {
        array[j] = array[j - current_gap];
        j -= current_gap;
      }

      array[j] = temp_element;
    }

    if (current_gap <= array_size / 2) {
      BatcherMergeSimpleIterative(array, 0, array_size - 1);
    }
  }

  BatcherMergeSimpleIterative(array, 0, array_size - 1);
}

}  // namespace bortsova_a_shell_batcher
