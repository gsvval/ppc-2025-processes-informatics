#pragma once

#include <tuple>

#include "task/include/task.hpp"

namespace guseva_a_matrix_sums {
using InType = std::tuple<uint32_t, uint32_t, std::vector<double>>;
using OutType = std::vector<double>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace guseva_a_matrix_sums
