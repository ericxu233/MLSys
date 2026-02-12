#ifndef GRAPH_H_
#define GRAPH_H_

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "problem_json.h"

namespace mlsys {

struct TensorNeighborhood {
  std::size_t tensor_id = 0;
  std::optional<std::size_t> producer_op;
  std::vector<std::size_t> user_ops;
  bool is_graph_input = false;
  bool is_graph_output = false;
};

class Graph {
 public:
  explicit Graph(const Problem& problem);

  const Problem& problem() const { return *problem_; }

  std::size_t NumTensors() const;
  std::size_t NumOps() const;

  const Tensor& GetTensor(std::size_t tensor_id) const;
  const Op& GetOp(std::size_t op_id) const;

  bool HasProducer(std::size_t tensor_id) const;
  std::optional<std::size_t> ProducerOp(std::size_t tensor_id) const;
  const std::vector<std::size_t>& UserOps(std::size_t tensor_id) const;

  bool IsGraphInputTensor(std::size_t tensor_id) const;
  bool IsGraphOutputTensor(std::size_t tensor_id) const;
  TensorNeighborhood DescribeTensor(std::size_t tensor_id) const;
  std::vector<TensorNeighborhood> DescribeAllTensors() const;
  const std::vector<std::size_t>& GraphInputTensors() const;
  const std::vector<std::size_t>& GraphOutputTensors() const;

  const std::vector<std::size_t>& PredecessorOps(std::size_t op_id) const;
  const std::vector<std::size_t>& SuccessorOps(std::size_t op_id) const;
  std::vector<std::size_t> TopologicalOrder() const;

 private:
  static constexpr std::size_t kInvalidOp = static_cast<std::size_t>(-1);

  void CheckTensorId(std::size_t tensor_id) const;
  void CheckOpId(std::size_t op_id) const;

  const Problem* problem_ = nullptr;
  std::vector<std::size_t> tensor_producer_op_;
  std::vector<std::vector<std::size_t>> tensor_user_ops_;
  std::vector<std::vector<std::size_t>> op_predecessors_;
  std::vector<std::vector<std::size_t>> op_successors_;
  std::vector<std::size_t> graph_input_tensors_;
  std::vector<std::size_t> graph_output_tensors_;
};

std::string BuildDot(const Graph& graph);
void WriteDotFile(const Graph& graph, const std::string& dot_path);

}  // namespace mlsys

#endif  // GRAPH_H_
