#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "problem_json.h"

namespace {

struct CliOptions {
  std::string input_path;
  bool emit_dot = false;
  std::string dot_path;
};

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

void WriteDot(const mlsys::Problem& problem, const std::string& dot_path) {
  std::ofstream out(dot_path);
  if (!out) {
    throw std::runtime_error("Failed to open output file: " + dot_path);
  }

  std::vector<int> produced(problem.tensors.size(), 0);
  std::vector<int> consumed(problem.tensors.size(), 0);
  for (const mlsys::Op& op : problem.ops) {
    for (std::size_t t : op.outputs) {
      ++produced[t];
    }
    for (std::size_t t : op.inputs) {
      ++consumed[t];
    }
  }

  out << "digraph MLSysGraph {\n";
  out << "  rankdir=LR;\n";
  out << "  graph [fontname=\"Helvetica\", splines=true, overlap=false];\n";
  out << "  node [fontname=\"Helvetica\", fontsize=10];\n";
  out << "  edge [fontname=\"Helvetica\", fontsize=9];\n";
  out << "\n";

  for (std::size_t t = 0; t < problem.tensors.size(); ++t) {
    const mlsys::Tensor& tensor = problem.tensors[t];
    const bool is_graph_input = produced[t] == 0;
    const bool is_graph_output = consumed[t] == 0;
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
    label << "Tensor[" << t << "]\\n" << tensor.height << "x" << tensor.width;
    if (is_graph_input) {
      label << "\\ninput";
    }
    if (is_graph_output) {
      label << "\\noutput";
    }

    out << "  t" << t
        << " [shape=ellipse, style=filled, fillcolor=\"" << fill_color
        << "\", color=\"" << border_color << "\", label=\""
        << EscapeDotString(label.str()) << "\"];\n";
  }

  out << "\n";
  for (std::size_t i = 0; i < problem.ops.size(); ++i) {
    const mlsys::Op& op = problem.ops[i];
    std::ostringstream label;
    label << "Op[" << i << "]\\n" << op.type << "\\ncost=" << op.base_cost;
    out << "  o" << i
        << " [shape=box, style=\"rounded,filled\", fillcolor=\"#fee2e2\", "
           "color=\"#991b1b\", label=\""
        << EscapeDotString(label.str()) << "\"];\n";
  }

  out << "\n";
  for (std::size_t i = 0; i < problem.ops.size(); ++i) {
    const mlsys::Op& op = problem.ops[i];
    for (std::size_t in_tensor : op.inputs) {
      out << "  t" << in_tensor << " -> o" << i << ";\n";
    }
    for (std::size_t out_tensor : op.outputs) {
      out << "  o" << i << " -> t" << out_tensor << ";\n";
    }
  }

  out << "}\n";
}

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
  std::cerr << "Usage: " << program_name << " <input.json> [--dot [output.dot]]\n";
  std::cerr << "  --dot           Emit graph as DOT. Uses derived path if no file is given.\n";
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
    CliOptions options = ParseArgs(argc, argv);
    const mlsys::Problem problem = mlsys::ReadProblemFromJson(options.input_path);

    std::cout << "Graph summary: " << problem.ops.size() << " ops, "
              << problem.tensors.size() << " tensors.\n";

    if (options.emit_dot) {
      WriteDot(problem, options.dot_path);
      std::cout << "DOT file written to: " << options.dot_path << "\n";
    } else {
      std::cout << "DOT output disabled. Pass --dot to emit graph output.\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    PrintUsage(argv[0]);
    return 1;
  }

  return 0;
}
