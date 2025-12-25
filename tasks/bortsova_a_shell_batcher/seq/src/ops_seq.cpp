#include "bortsova_a_shell_batcher/seq/include/ops_seq.hpp"

#include <vector>

#include "bortsova_a_shell_batcher/common/include/common.hpp"

namespace bortsova_a_shell_batcher {

BortsovaAShellBatcherkSEQ::BortsovaAShellBatcherkSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool BortsovaAShellBatcherkSEQ::ValidationImpl() {
  return true;
}

bool BortsovaAShellBatcherkSEQ::PreProcessingImpl() {
  return true;
}

bool BortsovaAShellBatcherkSEQ::RunImpl() {
  auto &array = GetInput();
  ShellSortBatcherMerge(array);
  GetOutput() = array;
  return true;
}

bool BortsovaAShellBatcherkSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace bortsova_a_shell_batcher
