#include <gtest/gtest.h>

#include <algorithm>

#include "bortsova_a_shell_batcher/common/include/common.hpp"
#include "bortsova_a_shell_batcher/mpi/include/ops_mpi.hpp"
#include "bortsova_a_shell_batcher/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace bortsova_a_shell_batcher {

class BortsovaAShellBatcherkPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 1'500'000;
  InType input_data_;

  void SetUp() override {
    input_data_.reserve(kCount_);
    for (int i = kCount_ - 1; i >= 0; i--) {
      input_data_.emplace_back(static_cast<double>(i));
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::ranges::is_sorted(output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(BortsovaAShellBatcherkPerfTests, Perf) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, BortsovaAShellBatcherkMPI, BortsovaAShellBatcherkSEQ>(
    PPC_SETTINGS_bortsova_a_shell_batcher);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BortsovaAShellBatcherkPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(BortsovaAShellPerf, BortsovaAShellBatcherkPerfTests, kGtestValues, kPerfTestName);

}  // namespace bortsova_a_shell_batcher
