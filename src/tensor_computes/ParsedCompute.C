/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ParsedCompute.h"

#include "MooseUtils.h"
#include "SwiftUtils.h"
#include "MultiMooseEnum.h"
#include "DomainAction.h"
#include "libmesh/fparser_ad.hh"

registerMooseObject("SwiftApp", ParsedCompute);

InputParameters
ParsedCompute::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("ParsedCompute object.");
  params.addRequiredParam<std::string>("expression", "Parsed expression");
  params.addParam<std::vector<TensorInputBufferName>>(
      "inputs", {}, "Buffer names used in the expression");
  params.addParam<std::vector<TensorInputBufferName>>(
      "derivatives", {}, "List of inputs to take the derivative w.r.t. (or none)");
  params.addParam<bool>("enable_fpoptimizer", true, "Use algebraic optimizer");
  params.addParam<bool>("extra_symbols",
                        false,
                        "Provide i (imaginary unit), kx,ky,kz (reciprocal space frequency), k2 "
                        "(square of the k-vector), x,y,z "
                        "(real space coordinates), time t, pi, and e.");
  // Constants and their values
  params.addParam<std::vector<std::string>>(
      "constant_names",
      std::vector<std::string>(),
      "Vector of constants used in the parsed function (use this for kB etc.)");
  params.addParam<std::vector<std::string>>(
      "constant_expressions",
      std::vector<std::string>(),
      "Vector of values for the constants in constant_names (can be an FParser expression)");
  MooseEnum expandEnum("REAL RECIPROCAL NONE", "NONE");
  params.addParam<MooseEnum>("expand", expandEnum, "Expand the tensor to full size.");
  return params;
}

ParsedCompute::ParsedCompute(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _extra_symbols(getParam<bool>("extra_symbols")),
    _expand(getParam<MooseEnum>("expand").getEnum<ExpandEnum>())
{
  const auto & expression = getParam<std::string>("expression");
  const auto & names = getParam<std::vector<TensorInputBufferName>>("inputs");

  // check for duplicates
  auto hasDuplicates = [](const std::vector<std::string> & values)
  {
    std::set<std::string> s(values.begin(), values.end());
    return values.size() != s.size();
  };

  if (hasDuplicates(names))
    paramError("inputs", "Duplicate buffer name.");

  // get all input buffers
  for (const auto & name : names)
    _params.push_back(&getInputBufferByName(name));

  static const std::vector<std::string> reserved_symbols = {
      "i", "x", "kx", "y", "ky", "z", "kz", "k2", "t", "pi", "e"};

  // helper function to check if the name given is one of the reserved_names
  auto isReservedName = [this](const auto & name)
  { return _extra_symbols && std::count(reserved_symbols.begin(), reserved_symbols.end(), name); };

  const auto & constant_names = getParam<std::vector<std::string>>("constant_names");
  const auto & constant_expressions = getParam<std::vector<std::string>>("constant_expressions");

  if (hasDuplicates(constant_names))
    paramError("constant_names", "Duplicate constant name.");

  for (const auto & name : constant_names)
    if (isReservedName(name))
      paramError("constant_names", "Cannot use reserved name '", name, "' for constant.");
  for (const auto & name : names)
    if (isReservedName(name))
      paramError("inputs", "Cannot use reserved name '", name, "' for coupled fields.");

  // check constant vectors
  unsigned int nconst = constant_expressions.size();
  if (nconst != constant_names.size())
    paramError("constant_names",
               "The parameter vectors constant_names (size ",
               constant_names.size(),
               ") and constant_values (size ",
               nconst,
               ") must have equal length.");

  // Build the expression with constants substituted
  std::string processed_expr = expression;

  // Evaluate constants
  std::vector<Real> constant_values(nconst);
  for (unsigned int i = 0; i < nconst; ++i)
  {
    // no need to use dual numbers for the constant expressions
    auto expr = std::make_shared<FunctionParserADBase<Real>>();

    // add previously evaluated constants
    for (unsigned int j = 0; j < i; ++j)
      if (!expr->AddConstant(constant_names[j], constant_values[j]))
        paramError("constant_names", "Invalid constant name '", constant_names[j], "'");

    // build the temporary constant expression function
    if (expr->Parse(constant_expressions[i], "") >= 0)
      mooseError("Invalid constant expression\n",
                 constant_expressions[i],
                 "\n in parsed function object.\n",
                 expr->ErrorMsg());

    constant_values[i] = expr->Eval(nullptr);
  }

  // Pre-substitute pi and e in the expression (they are mathematical constants, not variables)
  if (_extra_symbols)
  {
    // Replace 'pi' with its numeric value
    {
      std::string pi_str = std::to_string(libMesh::pi);
      size_t pos = 0;
      while ((pos = processed_expr.find("pi", pos)) != std::string::npos)
      {
        // Check if this is a complete identifier
        bool is_complete = true;
        if (pos > 0)
        {
          char prev = processed_expr[pos - 1];
          if (std::isalnum(prev) || prev == '_')
            is_complete = false;
        }
        if (pos + 2 < processed_expr.length())
        {
          char next = processed_expr[pos + 2];
          if (std::isalnum(next) || next == '_')
            is_complete = false;
        }

        if (is_complete)
        {
          processed_expr.replace(pos, 2, "(" + pi_str + ")");
          pos += pi_str.length() + 2;
        }
        else
          pos += 2;
      }
    }

    // Replace standalone 'e' (but not in scientific notation like 1e-5)
    {
      std::string e_str = std::to_string(std::exp(Real(1.0)));
      size_t pos = 0;
      while ((pos = processed_expr.find("e", pos)) != std::string::npos)
      {
        // Check if this is a complete identifier (not part of scientific notation)
        bool is_complete = true;
        if (pos > 0)
        {
          char prev = processed_expr[pos - 1];
          // If previous char is a digit, this might be scientific notation
          if (std::isalnum(prev) || prev == '_')
            is_complete = false;
        }
        if (pos + 1 < processed_expr.length())
        {
          char next = processed_expr[pos + 1];
          // If next char is +, -, or digit, this is scientific notation
          if (std::isalnum(next) || next == '_' || next == '+' || next == '-')
            is_complete = false;
        }

        if (is_complete)
        {
          processed_expr.replace(pos, 1, "(" + e_str + ")");
          pos += e_str.length() + 2;
        }
        else
          pos += 1;
      }
    }
  }

  // Build list of variables (excluding pi and e which are now substituted)
  std::vector<std::string> variables_vec = names;

  // add extra symbols (but not pi and e)
  if (_extra_symbols)
  {
    // Only include the actual tensor variables, not mathematical constants
    static const std::vector<std::string> tensor_symbols = {"i", "x", "kx", "y", "ky", "z", "kz", "k2", "t"};
    variables_vec.insert(variables_vec.end(), tensor_symbols.begin(), tensor_symbols.end());

    _constant_tensors.push_back(
        torch::tensor(c10::complex<double>(0.0, 1.0), MooseTensor::complexFloatTensorOptions()));
    _params.push_back(&_constant_tensors[0]);

    for (const auto dim : make_range(3u))
    {
      _params.push_back(&_domain.getAxis(dim));
      _params.push_back(&_domain.getReciprocalAxis(dim));
    }

    _params.push_back(&_domain.getKSquare());
    _params.push_back(&_time_tensor);
  }

  // Substitute constants in expression
  for (unsigned int i = 0; i < nconst; ++i)
  {
    std::string placeholder = constant_names[i];
    std::string value = std::to_string(constant_values[i]);

    size_t pos = 0;
    while ((pos = processed_expr.find(placeholder, pos)) != std::string::npos)
    {
      // Check if this is a complete identifier (not part of a larger identifier)
      bool is_complete = true;
      if (pos > 0)
      {
        char prev = processed_expr[pos - 1];
        if (std::isalnum(prev) || prev == '_')
          is_complete = false;
      }
      if (pos + placeholder.length() < processed_expr.length())
      {
        char next = processed_expr[pos + placeholder.length()];
        if (std::isalnum(next) || next == '_')
          is_complete = false;
      }

      if (is_complete)
        processed_expr.replace(pos, placeholder.length(), value);
      else
        pos += placeholder.length();
    }
  }

  // Parse the expression
  if (!_parser.parse(processed_expr, variables_vec))
    paramError("expression", "Invalid function: ", _parser.errorMessage());

  // Take derivatives
  for (const auto & d : getParam<std::vector<TensorInputBufferName>>("derivatives"))
  {
    if (std::find(names.begin(), names.end(), d) != names.end())
      _parser.differentiate(d);
    else
      paramError("derivatives",
                 "Derivative w.r.t `",
                 d,
                 "` was requested, but it is not listed in `inputs`.");
  }

  // Compile (optimization is done during compilation if enable_fpoptimizer is true)
  // The compile step performs AST simplification and JIT graph optimization
  _parser.compile();
}

void
ParsedCompute::computeBuffer()
{
  if (_extra_symbols)
  {
    _time_tensor = torch::tensor(_time, MooseTensor::floatTensorOptions());

    // Update constants
    _constant_tensors[0] = torch::tensor(c10::complex<double>(0.0, 1.0),
                                         MooseTensor::complexFloatTensorOptions());
  }

  // Evaluate using JIT-compiled graph
  _u = _parser.eval(_params);

  // optionally expand the tensor
  switch (_expand)
  {
    case ExpandEnum::REAL:
      _u = _u.expand(_domain.getShape());
      break;

    case ExpandEnum::RECIPROCAL:
      _u = _u.expand(_domain.getReciprocalShape());
      break;

    case ExpandEnum::NONE:
      break;
  }
}
