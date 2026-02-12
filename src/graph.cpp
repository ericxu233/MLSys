#include "graph.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
void SortAndUnique(std::vector<T>* values) {
  std::sort(values->begin(), values->end());
  values->erase(std::unique(values->begin(), values->end()), values->end());
}

std::string EscapeDotString(const std::string& raw) {
  std::string out;
  out.reserve(raw.size() + 16);
  for (char c : raw) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      default:
        out.push_back(c);
        break;
    }
  }
  return out;
}

}  // namespace

namespace mlsys {

Graph::Graph(const Problem& problem) : problem_(&problem) {
  tensor_producer_op_.assign(problem_->tensors.size(), kInvalidOp);
  tensor_user_ops_.assign(problem_->tensors.size(), {});
  op_predecessors_.assign(problem_->ops.size(), {});
  op_successors_.assign(problem_->ops.size(), {});

  for (std::size_t op_id = 0; op_id < problem_->ops.size(); ++op_id) {
    const Op& op = problem_->ops[op_id];

    for (std::size_t tensor_id : op.inputs) {
      CheckTensorId(tensor_id);
      tensor_user_ops_[tensor_id].push_back(op_id);
    }

    for (std::size_t tensor_id : op.outputs) {
      CheckTensorId(tensor_id);
      if (tensor_producer_op_[tensor_id] != kInvalidOp &&
          tensor_producer_op_[tensor_id] != op_id) {
        std::ostringstream oss;
        oss << "Tensor[" << tensor_id << "] has multiple producers: Op["
            << tensor_producer_op_[tensor_id] << "] and Op[" << op_id << "].";
        throw std::runtime_error(oss.str());
      }
      tensor_producer_op_[tensor_id] = op_id;
    }
  }

  for (std::vector<std::size_t>& users : tensor_user_ops_) {
    SortAndUnique(&users);
  }

  for (std::size_t op_id = 0; op_id < problem_->ops.size(); ++op_id) {
    std::vector<std::size_t> predecessors;
    for (std::size_t tensor_id : problem_->ops[op_id].inputs) {
      const std::size_t producer = tensor_producer_op_[tensor_id];
      if (producer != kInvalidOp && producer != op_id) {
        predecessors.push_back(producer);
      }
    }
    SortAndUnique(&predecessors);
    op_predecessors_[op_id] = predecessors;
  }

  for (std::size_t op_id = 0; op_id < problem_->ops.size(); ++op_id) {
    for (std::size_t predecessor : op_predecessors_[op_id]) {
      op_successors_[predecessor].push_back(op_id);
    }
  }
  for (std::vector<std::size_t>& successors : op_successors_) {
    SortAndUnique(&successors);
  }

  graph_input_tensors_.reserve(problem_->tensors.size());
  graph_output_tensors_.reserve(problem_->tensors.size());
  for (std::size_t tensor_id = 0; tensor_id < problem_->tensors.size();
       ++tensor_id) {
    if (IsGraphInputTensor(tensor_id)) {
      graph_input_tensors_.push_back(tensor_id);
    }
    if (IsGraphOutputTensor(tensor_id)) {
      graph_output_tensors_.push_back(tensor_id);
    }
  }
}

std::size_t Graph::NumTensors() const { return problem_->tensors.size(); }

std::size_t Graph::NumOps() const { return problem_->ops.size(); }

const Tensor& Graph::GetTensor(std::size_t tensor_id) const {
  CheckTensorId(tensor_id);
  return problem_->tensors[tensor_id];
}

const Op& Graph::GetOp(std::size_t op_id) const {
  CheckOpId(op_id);
  return problem_->ops[op_id];
}

bool Graph::HasProducer(std::size_t tensor_id) const {
  CheckTensorId(tensor_id);
  return tensor_producer_op_[tensor_id] != kInvalidOp;
}

std::optional<std::size_t> Graph::ProducerOp(std::size_t tensor_id) const {
  CheckTensorId(tensor_id);
  if (tensor_producer_op_[tensor_id] == kInvalidOp) {
    return std::nullopt;
  }
  return tensor_producer_op_[tensor_id];
}

const std::vector<std::size_t>& Graph::UserOps(std::size_t tensor_id) const {
  CheckTensorId(tensor_id);
  return tensor_user_ops_[tensor_id];
}

bool Graph::IsGraphInputTensor(std::size_t tensor_id) const {
  return !HasProducer(tensor_id);
}

bool Graph::IsGraphOutputTensor(std::size_t tensor_id) const {
  return UserOps(tensor_id).empty();
}

TensorNeighborhood Graph::DescribeTensor(std::size_t tensor_id) const {
  const std::optional<std::size_t> producer = ProducerOp(tensor_id);
  TensorNeighborhood summary;
  summary.tensor_id = tensor_id;
  summary.producer_op = producer;
  summary.user_ops = UserOps(tensor_id);
  summary.is_graph_input = !producer.has_value();
  summary.is_graph_output = UserOps(tensor_id).empty();
  return summary;
}

std::vector<TensorNeighborhood> Graph::DescribeAllTensors() const {
  std::vector<TensorNeighborhood> summaries;
  summaries.reserve(NumTensors());
  for (std::size_t tensor_id = 0; tensor_id < NumTensors(); ++tensor_id) {
    summaries.push_back(DescribeTensor(tensor_id));
  }
  return summaries;
}

const std::vector<std::size_t>& Graph::GraphInputTensors() const {
  return graph_input_tensors_;
}

const std::vector<std::size_t>& Graph::GraphOutputTensors() const {
  return graph_output_tensors_;
}

const std::vector<std::size_t>& Graph::PredecessorOps(std::size_t op_id) const {
  CheckOpId(op_id);
  return op_predecessors_[op_id];
}

const std::vector<std::size_t>& Graph::SuccessorOps(std::size_t op_id) const {
  CheckOpId(op_id);
  return op_successors_[op_id];
}

std::vector<std::size_t> Graph::TopologicalOrder() const {
  std::vector<std::size_t> indegree(NumOps(), 0);
  for (std::size_t op_id = 0; op_id < NumOps(); ++op_id) {
    indegree[op_id] = op_predecessors_[op_id].size();
  }

  std::queue<std::size_t> ready;
  for (std::size_t op_id = 0; op_id < NumOps(); ++op_id) {
    if (indegree[op_id] == 0) {
      ready.push(op_id);
    }
  }

  std::vector<std::size_t> order;
  order.reserve(NumOps());
  while (!ready.empty()) {
    const std::size_t op_id = ready.front();
    ready.pop();
    order.push_back(op_id);

    for (std::size_t successor : op_successors_[op_id]) {
      if (--indegree[successor] == 0) {
        ready.push(successor);
      }
    }
  }

  if (order.size() != NumOps()) {
    throw std::runtime_error("Op graph is not a DAG: topological sort failed.");
  }
  return order;
}

void Graph::CheckTensorId(std::size_t tensor_id) const {
  if (tensor_id >= problem_->tensors.size()) {
    std::ostringstream oss;
    oss << "Tensor id out of range: " << tensor_id;
    throw std::runtime_error(oss.str());
  }
}

void Graph::CheckOpId(std::size_t op_id) const {
  if (op_id >= problem_->ops.size()) {
    std::ostringstream oss;
    oss << "Op id out of range: " << op_id;
    throw std::runtime_error(oss.str());
  }
}

std::string BuildDot(const Graph& graph) {
  std::ostringstream out;

  out << "digraph MLSysGraph {\n";
  out << "  rankdir=LR;\n";
  out << "  graph [fontname=\"Helvetica\", splines=true, overlap=false];\n";
  out << "  node [fontname=\"Helvetica\", fontsize=10];\n";
  out << "  edge [fontname=\"Helvetica\", fontsize=9];\n";
  out << "\n";

  for (std::size_t tensor_id = 0; tensor_id < graph.NumTensors(); ++tensor_id) {
    const Tensor& tensor = graph.GetTensor(tensor_id);
    const bool is_graph_input = graph.IsGraphInputTensor(tensor_id);
    const bool is_graph_output = graph.IsGraphOutputTensor(tensor_id);

    std::string fill_color = "#f3f4f6";
    std::string border_color = "#6b7280";
    if (is_graph_input && is_graph_output) {
      fill_color = "#fde68a";
      border_color = "#b45309";
    } else if (is_graph_input) {
      fill_color = "#bbf7d0";
      border_color = "#15803d";
    } else if (is_graph_output) {
      fill_color = "#bfdbfe";
      border_color = "#1d4ed8";
    }

    std::ostringstream label;
    label << "Tensor[" << tensor_id << "]\\n" << tensor.height << "x"
          << tensor.width;
    if (is_graph_input) {
      label << "\\ninput";
    }
    if (is_graph_output) {
      label << "\\noutput";
    }

    out << "  t" << tensor_id
        << " [shape=ellipse, style=filled, fillcolor=\"" << fill_color
        << "\", color=\"" << border_color << "\", label=\""
        << EscapeDotString(label.str()) << "\"];\n";
  }

  out << "\n";
  for (std::size_t op_id = 0; op_id < graph.NumOps(); ++op_id) {
    const Op& op = graph.GetOp(op_id);
    std::ostringstream label;
    label << "Op[" << op_id << "]\\n" << op.type << "\\ncost=" << op.base_cost
          << "\\npreds=" << graph.PredecessorOps(op_id).size()
          << "\\nsuccs=" << graph.SuccessorOps(op_id).size();

    out << "  o" << op_id
        << " [shape=box, style=\"rounded,filled\", fillcolor=\"#fee2e2\", "
           "color=\"#991b1b\", label=\""
        << EscapeDotString(label.str()) << "\"];\n";
  }

  out << "\n";
  for (std::size_t op_id = 0; op_id < graph.NumOps(); ++op_id) {
    const Op& op = graph.GetOp(op_id);
    for (std::size_t input_tensor : op.inputs) {
      out << "  t" << input_tensor << " -> o" << op_id << ";\n";
    }
    for (std::size_t output_tensor : op.outputs) {
      out << "  o" << op_id << " -> t" << output_tensor << ";\n";
    }
  }

  out << "}\n";
  return out.str();
}

void WriteDotFile(const Graph& graph, const std::string& dot_path) {
  std::ofstream out(dot_path);
  if (!out) {
    throw std::runtime_error("Failed to open output file: " + dot_path);
  }
  out << BuildDot(graph);
}

}  // namespace mlsys
