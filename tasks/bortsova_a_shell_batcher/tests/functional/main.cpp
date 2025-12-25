#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "bortsova_a_shell_batcher/common/include/common.hpp"
#include "bortsova_a_shell_batcher/mpi/include/ops_mpi.hpp"
#include "bortsova_a_shell_batcher/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace bortsova_a_shell_batcher {

class BortsovaAShellBatcherkFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return "reversed_length_" + std::to_string(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::vector<double>(params);
    for (int i = params - 1; i >= 0; i--) {
      input_data_.emplace_back(static_cast<double>(i));
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::ranges::is_sorted(output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(BortsovaAShellBatcherkFuncTests, ShellBatcher) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 16> kTestParam = {0, 1, 2, 3, 4, 40, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<BortsovaAShellBatcherkMPI, InType>(kTestParam, PPC_SETTINGS_bortsova_a_shell_batcher),
    ppc::util::AddFuncTask<BortsovaAShellBatcherkSEQ, InType>(kTestParam, PPC_SETTINGS_bortsova_a_shell_batcher));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BortsovaAShellBatcherkFuncTests::PrintFuncTestName<BortsovaAShellBatcherkFuncTests>;

INSTANTIATE_TEST_SUITE_P(BortsovaAShellFunc, BortsovaAShellBatcherkFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace bortsova_a_shell_batcher
