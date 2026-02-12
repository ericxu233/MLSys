#include "problem_json.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct JsonValue {
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };

  using Array = std::vector<JsonValue>;
  using Object = std::unordered_map<std::string, JsonValue>;

  Type type = Type::kNull;
  bool bool_value = false;
  double number_value = 0.0;
  std::string string_value;
  std::shared_ptr<Array> array_value;
  std::shared_ptr<Object> object_value;

  static JsonValue MakeNull() { return JsonValue{}; }

  static JsonValue MakeBool(bool value) {
    JsonValue v;
    v.type = Type::kBool;
    v.bool_value = value;
    return v;
  }

  static JsonValue MakeNumber(double value) {
    JsonValue v;
    v.type = Type::kNumber;
    v.number_value = value;
    return v;
  }

  static JsonValue MakeString(std::string value) {
    JsonValue v;
    v.type = Type::kString;
    v.string_value = std::move(value);
    return v;
  }

  static JsonValue MakeArray(Array value) {
    JsonValue v;
    v.type = Type::kArray;
    v.array_value = std::make_shared<Array>(std::move(value));
    return v;
  }

  static JsonValue MakeObject(Object value) {
    JsonValue v;
    v.type = Type::kObject;
    v.object_value = std::make_shared<Object>(std::move(value));
    return v;
  }

  const Array& AsArray() const {
    if (type != Type::kArray) {
      throw std::runtime_error("JSON type error: expected array.");
    }
    return *array_value;
  }

  const Object& AsObject() const {
    if (type != Type::kObject) {
      throw std::runtime_error("JSON type error: expected object.");
    }
    return *object_value;
  }

  const std::string& AsString() const {
    if (type != Type::kString) {
      throw std::runtime_error("JSON type error: expected string.");
    }
    return string_value;
  }

  double AsNumber() const {
    if (type != Type::kNumber) {
      throw std::runtime_error("JSON type error: expected number.");
    }
    return number_value;
  }
};

class JsonParser {
 public:
  explicit JsonParser(std::string text) : text_(std::move(text)) {}

  JsonValue Parse() {
    SkipWhitespace();
    JsonValue value = ParseValue();
    SkipWhitespace();
    if (pos_ != text_.size()) {
      throw Error("Unexpected trailing characters.");
    }
    return value;
  }

 private:
  static bool IsDigit(char c) { return c >= '0' && c <= '9'; }

  [[noreturn]] std::runtime_error Error(const std::string& msg) const {
    std::ostringstream oss;
    oss << "JSON parse error at offset " << pos_ << ": " << msg;
    throw std::runtime_error(oss.str());
  }

  void SkipWhitespace() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_])) != 0) {
      ++pos_;
    }
  }

  bool MatchLiteral(std::string_view literal) {
    if (text_.compare(pos_, literal.size(), literal) == 0) {
      pos_ += literal.size();
      return true;
    }
    return false;
  }

  char Peek() const {
    if (pos_ >= text_.size()) {
      throw Error("Unexpected end of input.");
    }
    return text_[pos_];
  }

  char Get() {
    const char c = Peek();
    ++pos_;
    return c;
  }

  void Expect(char c) {
    if (Get() != c) {
      std::ostringstream oss;
      oss << "Expected '" << c << "'.";
      throw Error(oss.str());
    }
  }

  JsonValue ParseValue() {
    SkipWhitespace();
    const char c = Peek();
    switch (c) {
      case 'n':
        return ParseNull();
      case 't':
      case 'f':
        return ParseBool();
      case '"':
        return JsonValue::MakeString(ParseString());
      case '[':
        return ParseArray();
      case '{':
        return ParseObject();
      default:
        if (c == '-' || IsDigit(c)) {
          return JsonValue::MakeNumber(ParseNumber());
        }
        throw Error("Invalid value.");
    }
  }

  JsonValue ParseNull() {
    if (!MatchLiteral("null")) {
      throw Error("Invalid literal, expected 'null'.");
    }
    return JsonValue::MakeNull();
  }

  JsonValue ParseBool() {
    if (MatchLiteral("true")) {
      return JsonValue::MakeBool(true);
    }
    if (MatchLiteral("false")) {
      return JsonValue::MakeBool(false);
    }
    throw Error("Invalid literal, expected 'true' or 'false'.");
  }

  void AppendCodepointUtf8(unsigned codepoint, std::string* out) {
    if (codepoint <= 0x7F) {
      out->push_back(static_cast<char>(codepoint));
      return;
    }
    if (codepoint <= 0x7FF) {
      out->push_back(static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F)));
      out->push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
      return;
    }
    out->push_back(static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F)));
    out->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  }

  unsigned ParseHex4() {
    if (pos_ + 4 > text_.size()) {
      throw Error("Incomplete \\u escape.");
    }
    unsigned value = 0;
    for (int i = 0; i < 4; ++i) {
      value <<= 4;
      const char c = text_[pos_++];
      if (c >= '0' && c <= '9') {
        value |= static_cast<unsigned>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        value |= static_cast<unsigned>(10 + c - 'a');
      } else if (c >= 'A' && c <= 'F') {
        value |= static_cast<unsigned>(10 + c - 'A');
      } else {
        throw Error("Invalid hex digit in \\u escape.");
      }
    }
    return value;
  }

  std::string ParseString() {
    Expect('"');
    std::string out;
    while (true) {
      if (pos_ >= text_.size()) {
        throw Error("Unterminated string.");
      }
      const char c = Get();
      if (c == '"') {
        break;
      }
      if (c == '\\') {
        if (pos_ >= text_.size()) {
          throw Error("Invalid escape sequence.");
        }
        const char esc = Get();
        switch (esc) {
          case '"':
          case '\\':
          case '/':
            out.push_back(esc);
            break;
          case 'b':
            out.push_back('\b');
            break;
          case 'f':
            out.push_back('\f');
            break;
          case 'n':
            out.push_back('\n');
            break;
          case 'r':
            out.push_back('\r');
            break;
          case 't':
            out.push_back('\t');
            break;
          case 'u': {
            const unsigned cp = ParseHex4();
            AppendCodepointUtf8(cp, &out);
            break;
          }
          default:
            throw Error("Unsupported escape sequence.");
        }
      } else {
        out.push_back(c);
      }
    }
    return out;
  }

  double ParseNumber() {
    const std::size_t start = pos_;
    if (Peek() == '-') {
      ++pos_;
    }
    if (pos_ >= text_.size()) {
      throw Error("Invalid number.");
    }
    if (text_[pos_] == '0') {
      ++pos_;
    } else {
      if (!IsDigit(text_[pos_])) {
        throw Error("Invalid number.");
      }
      while (pos_ < text_.size() && IsDigit(text_[pos_])) {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      if (pos_ >= text_.size() || !IsDigit(text_[pos_])) {
        throw Error("Invalid number.");
      }
      while (pos_ < text_.size() && IsDigit(text_[pos_])) {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
        ++pos_;
      }
      if (pos_ >= text_.size() || !IsDigit(text_[pos_])) {
        throw Error("Invalid number.");
      }
      while (pos_ < text_.size() && IsDigit(text_[pos_])) {
        ++pos_;
      }
    }
    const std::string token = text_.substr(start, pos_ - start);
    char* endptr = nullptr;
    const double value = std::strtod(token.c_str(), &endptr);
    if (endptr == token.c_str() || *endptr != '\0') {
      throw Error("Failed to parse numeric token.");
    }
    return value;
  }

  JsonValue ParseArray() {
    Expect('[');
    SkipWhitespace();
    JsonValue::Array arr;
    if (Peek() == ']') {
      ++pos_;
      return JsonValue::MakeArray(std::move(arr));
    }
    while (true) {
      arr.push_back(ParseValue());
      SkipWhitespace();
      const char c = Get();
      if (c == ']') {
        break;
      }
      if (c != ',') {
        throw Error("Expected ',' or ']'.");
      }
      SkipWhitespace();
    }
    return JsonValue::MakeArray(std::move(arr));
  }

  JsonValue ParseObject() {
    Expect('{');
    SkipWhitespace();
    JsonValue::Object obj;
    if (Peek() == '}') {
      ++pos_;
      return JsonValue::MakeObject(std::move(obj));
    }
    while (true) {
      if (Peek() != '"') {
        throw Error("Expected string key.");
      }
      const std::string key = ParseString();
      SkipWhitespace();
      Expect(':');
      SkipWhitespace();
      obj[key] = ParseValue();
      SkipWhitespace();
      const char c = Get();
      if (c == '}') {
        break;
      }
      if (c != ',') {
        throw Error("Expected ',' or '}'.");
      }
      SkipWhitespace();
    }
    return JsonValue::MakeObject(std::move(obj));
  }

  std::string text_;
  std::size_t pos_ = 0;
};

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Failed to open input file: " + path);
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

const JsonValue& GetRequiredField(const JsonValue::Object& obj,
                                  const std::string& key) {
  const auto it = obj.find(key);
  if (it == obj.end()) {
    throw std::runtime_error("Missing required field: " + key);
  }
  return it->second;
}

int64_t ToInt64(const JsonValue& value, const std::string& field) {
  const double n = value.AsNumber();
  const double min_val =
      static_cast<double>(std::numeric_limits<int64_t>::min());
  const double max_val =
      static_cast<double>(std::numeric_limits<int64_t>::max());
  if (n < min_val || n > max_val) {
    throw std::runtime_error("Integer out of range in field: " + field);
  }
  const int64_t as_int = static_cast<int64_t>(n);
  if (static_cast<double>(as_int) != n) {
    throw std::runtime_error("Expected integer in field: " + field);
  }
  return as_int;
}

std::vector<int64_t> ToInt64Array(const JsonValue& value,
                                  const std::string& field) {
  const auto& arr = value.AsArray();
  std::vector<int64_t> out;
  out.reserve(arr.size());
  for (const JsonValue& element : arr) {
    out.push_back(ToInt64(element, field));
  }
  return out;
}

std::vector<std::string> ToStringArray(const JsonValue& value,
                                       const std::string& field) {
  const auto& arr = value.AsArray();
  std::vector<std::string> out;
  out.reserve(arr.size());
  for (const JsonValue& element : arr) {
    try {
      out.push_back(element.AsString());
    } catch (const std::runtime_error&) {
      throw std::runtime_error("Expected string in field: " + field);
    }
  }
  return out;
}

std::vector<std::vector<std::size_t>> ToIndex2DArray(const JsonValue& value,
                                                     const std::string& field) {
  const auto& outer = value.AsArray();
  std::vector<std::vector<std::size_t>> out;
  out.reserve(outer.size());
  for (const JsonValue& row : outer) {
    const auto& in = row.AsArray();
    std::vector<std::size_t> parsed_row;
    parsed_row.reserve(in.size());
    for (const JsonValue& element : in) {
      const int64_t idx = ToInt64(element, field);
      if (idx < 0) {
        throw std::runtime_error("Negative index in field: " + field);
      }
      parsed_row.push_back(static_cast<std::size_t>(idx));
    }
    out.push_back(std::move(parsed_row));
  }
  return out;
}

}  // namespace

namespace mlsys {

Problem ReadProblemFromJson(const std::string& path) {
  JsonParser parser(ReadFile(path));
  const JsonValue root = parser.Parse();
  const auto& obj = root.AsObject();

  const std::vector<int64_t> widths =
      ToInt64Array(GetRequiredField(obj, "widths"), "widths");
  const std::vector<int64_t> heights =
      ToInt64Array(GetRequiredField(obj, "heights"), "heights");
  if (widths.size() != heights.size()) {
    throw std::runtime_error("widths and heights must have identical length.");
  }

  const std::vector<std::string> op_types =
      ToStringArray(GetRequiredField(obj, "op_types"), "op_types");
  const std::vector<std::vector<std::size_t>> inputs =
      ToIndex2DArray(GetRequiredField(obj, "inputs"), "inputs");
  const std::vector<std::vector<std::size_t>> outputs =
      ToIndex2DArray(GetRequiredField(obj, "outputs"), "outputs");
  const std::vector<int64_t> base_costs =
      ToInt64Array(GetRequiredField(obj, "base_costs"), "base_costs");

  const std::size_t op_count = op_types.size();
  if (inputs.size() != op_count || outputs.size() != op_count ||
      base_costs.size() != op_count) {
    throw std::runtime_error(
        "op_types, inputs, outputs, and base_costs must have identical length.");
  }

  Problem problem;
  problem.tensors.reserve(widths.size());
  for (std::size_t i = 0; i < widths.size(); ++i) {
    problem.tensors.push_back(Tensor{widths[i], heights[i]});
  }

  problem.ops.reserve(op_count);
  for (std::size_t i = 0; i < op_count; ++i) {
    Op op;
    op.type = op_types[i];
    op.inputs = inputs[i];
    op.outputs = outputs[i];
    op.base_cost = base_costs[i];
    problem.ops.push_back(std::move(op));
  }

  problem.fast_memory_capacity =
      ToInt64(GetRequiredField(obj, "fast_memory_capacity"),
              "fast_memory_capacity");
  problem.slow_memory_bandwidth =
      ToInt64(GetRequiredField(obj, "slow_memory_bandwidth"),
              "slow_memory_bandwidth");

  const std::vector<int64_t> native_granularity =
      ToInt64Array(GetRequiredField(obj, "native_granularity"),
                   "native_granularity");
  if (native_granularity.size() != 2) {
    throw std::runtime_error("native_granularity must have exactly 2 entries.");
  }
  problem.native_width = native_granularity[0];
  problem.native_height = native_granularity[1];

  for (std::size_t op_id = 0; op_id < problem.ops.size(); ++op_id) {
    for (std::size_t tensor_id : problem.ops[op_id].inputs) {
      if (tensor_id >= problem.tensors.size()) {
        std::ostringstream oss;
        oss << "inputs[" << op_id << "] references invalid tensor id "
            << tensor_id;
        throw std::runtime_error(oss.str());
      }
    }
    for (std::size_t tensor_id : problem.ops[op_id].outputs) {
      if (tensor_id >= problem.tensors.size()) {
        std::ostringstream oss;
        oss << "outputs[" << op_id << "] references invalid tensor id "
            << tensor_id;
        throw std::runtime_error(oss.str());
      }
    }
  }

  return problem;
}

}  // namespace mlsys
