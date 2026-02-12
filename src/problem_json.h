#ifndef PROBLEM_JSON_H_
#define PROBLEM_JSON_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mlsys {

struct Tensor {
  int64_t width = 0;
  int64_t height = 0;
};

struct Op {
  std::string type;
  std::vector<std::size_t> inputs;
  std::vector<std::size_t> outputs;
  int64_t base_cost = 0;
};

struct Problem {
  std::vector<Tensor> tensors;
  std::vector<Op> ops;
  int64_t fast_memory_capacity = 0;
  int64_t slow_memory_bandwidth = 0;
  int64_t native_width = 0;
  int64_t native_height = 0;
};

Problem ReadProblemFromJson(const std::string& path);

}  // namespace mlsys

#endif  // PROBLEM_JSON_H_
