#pragma once

#include "bortsova_a_shell_batcher/common/include/common.hpp"
#include "task/include/task.hpp"

namespace bortsova_a_shell_batcher {

class BortsovaAShellBatcherkSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BortsovaAShellBatcherkSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace bortsova_a_shell_batcher
