#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "graph.h"
#include "problem_json.h"

namespace {

struct CliOptions {
  std::string input_path;
  bool emit_dot = false;
  std::string dot_path;
  bool show_tensor_info = false;
};

std::string DeriveDotPath(const std::string& input_path) {
  const std::size_t slash = input_path.find_last_of("/\\");
  const std::size_t dot = input_path.find_last_of('.');
  if (dot == std::string::npos ||
      (slash != std::string::npos && dot < slash + 1)) {
    return input_path + ".dot";
  }
  return input_path.substr(0, dot) + ".dot";
}

void PrintUsage(const char* program_name) {
  std::cerr << "Usage: " << program_name
            << " <input.json> [--dot [output.dot]] [--tensor-info]\n";
  std::cerr << "  --dot           Emit graph as DOT. Uses derived path if no file is given.\n";
  std::cerr << "  --tensor-info   Print producer/user mapping for every tensor.\n";
}

void PrintTensorInfo(const mlsys::Graph& graph) {
  for (const mlsys::TensorNeighborhood& info : graph.DescribeAllTensors()) {
    const mlsys::Tensor& tensor = graph.GetTensor(info.tensor_id);

    std::cout << "Tensor[" << info.tensor_id << "] " << tensor.height << "x"
              << tensor.width << " producer=";
    if (info.producer_op.has_value()) {
      std::cout << "Op[" << *info.producer_op << "]";
    } else {
      std::cout << "<graph-input>";
    }

    std::cout << " users=[";
    for (std::size_t i = 0; i < info.user_ops.size(); ++i) {
      if (i > 0) {
        std::cout << ", ";
      }
      std::cout << "Op[" << info.user_ops[i] << "]";
    }
    std::cout << "]";

    if (info.is_graph_input) {
      std::cout << " input";
    }
    if (info.is_graph_output) {
      std::cout << " output";
    }
    std::cout << "\n";
  }
}

CliOptions ParseArgs(int argc, char** argv) {
  if (argc == 2) {
    const std::string only_arg = argv[1];
    if (only_arg == "--help" || only_arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    }
  }

  if (argc < 2) {
    throw std::runtime_error("Missing required input.json argument.");
  }

  CliOptions options;
  options.input_path = argv[1];

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    }

    if (arg == "--dot") {
      options.emit_dot = true;
      if (i + 1 < argc) {
        const std::string maybe_path = argv[i + 1];
        if (maybe_path.rfind("--", 0) != 0) {
          options.dot_path = maybe_path;
          ++i;
        }
      }
      continue;
    }

    if (arg == "--tensor-info") {
      options.show_tensor_info = true;
      continue;
    }

    constexpr const char kDotPrefix[] = "--dot=";
    if (arg.rfind(kDotPrefix, 0) == 0) {
      options.emit_dot = true;
      options.dot_path = arg.substr(sizeof(kDotPrefix) - 1);
      if (options.dot_path.empty()) {
        throw std::runtime_error("Expected non-empty path in --dot=<path>.");
      }
      continue;
    }

    throw std::runtime_error("Unknown argument: " + arg);
  }

  if (options.emit_dot && options.dot_path.empty()) {
    options.dot_path = DeriveDotPath(options.input_path);
  }

  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions options = ParseArgs(argc, argv);
    const mlsys::Problem problem = mlsys::ReadProblemFromJson(options.input_path);
    const mlsys::Graph graph(problem);
    const std::vector<std::size_t> topo_order = graph.TopologicalOrder();

    std::cout << "Graph summary: " << graph.NumOps() << " ops, "
              << graph.NumTensors() << " tensors, "
              << graph.GraphInputTensors().size() << " graph inputs, "
              << graph.GraphOutputTensors().size() << " graph outputs.\n";
    if (!topo_order.empty()) {
      std::cout << "Topological order starts at Op[" << topo_order.front()
                << "] and ends at Op[" << topo_order.back() << "].\n";
    }

    if (options.emit_dot) {
      mlsys::WriteDotFile(graph, options.dot_path);
      std::cout << "DOT file written to: " << options.dot_path << "\n";
    }
    if (options.show_tensor_info) {
      PrintTensorInfo(graph);
    }
    if (!options.emit_dot && !options.show_tensor_info) {
      std::cout << "DOT output disabled. Pass --dot to emit graph output.\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    PrintUsage(argv[0]);
    return 1;
  }

  return 0;
}
