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

registerMooseObject("SwiftApp", ParsedCompute);

InputParameters
ParsedCompute::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.addClassDescription("ParsedCompute object.");
  params.addRequiredParam<std::string>("expression", "Parsed expression");
  params.addParam<std::vector<TensorInputBufferName>>(
      "inputs", {}, "Buffer names used in the expression");
  params.addParam<std::vector<TensorInputBufferName>>(
      "derivatives", {}, "List of inputs to take the derivative w.r.t. (or none)");
  params.addParam<bool>(
      "enable_jit", true, "Use operator fusion and just in time compilation (recommended on GPU)");
  params.addParam<bool>("enable_fpoptimizer", true, "Use algebraic optimizer");
  params.addParam<bool>("extra_symbols",
                        false,
                        "Provide i (imaginary unit), kx,ky,kz (reciprocal space frequency), k2 "
                        "(square of the k-vector), x,y,z "
                        "(real space coordinates), and pi,e.");
  // Constants and their values
  params.addParam<std::vector<std::string>>(
      "constant_names",
      std::vector<std::string>(),
      "Vector of constants used in the parsed function (use this for kB etc.)");
  params.addParam<std::vector<std::string>>(
      "constant_expressions",
      std::vector<std::string>(),
      "Vector of values for the constants in constant_names (can be an FParser expression)");

  return params;
}

ParsedCompute::ParsedCompute(const InputParameters & parameters)
  : TensorOperator(parameters),
    _use_jit(getParam<bool>("enable_jit")),
    _extra_symbols(getParam<bool>("extra_symbols"))
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
      "i", "x", "kx", "y", "ky", "z", "kz", "k2"};

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

  auto setup = [&](auto & fp)
  {
    std::vector variables_vec = names;

    // add extra symbols
    if (_extra_symbols)
    {
      // append extra symbols
      variables_vec.insert(variables_vec.end(), reserved_symbols.begin(), reserved_symbols.end());

      _constant_tensors.push_back(torch::tensor(c10::complex<double>(0.0, 1.0)));
      _params.push_back(&_constant_tensors[0]);

      for (const auto dim : make_range(3u))
      {
        _params.push_back(&_domain.getAxis(dim));
        _params.push_back(&_domain.getReciprocalAxis(dim));
      }

      _params.push_back(&_domain.getKSquare());

      fp.AddConstant("pi", libMesh::pi);
      fp.AddConstant("e", std::exp(Real(1.0)));
    }

    // previously evaluated constant_expressions may be used in following constant_expressions
    std::vector<Real> constant_values(nconst);
    for (unsigned int i = 0; i < nconst; ++i)
    {
      // no need to use dual numbers for the constant expressions
      auto expression = std::make_shared<FunctionParserADBase<Real>>();

      // add previously evaluated constants
      for (unsigned int j = 0; j < i; ++j)
        if (!expression->AddConstant(constant_names[j], constant_values[j]))
          paramError("constant_names", "Invalid constant name '", constant_names[j], "'");

      // build the temporary constant expression function
      if (expression->Parse(constant_expressions[i], "") >= 0)
        mooseError("Invalid constant expression\n",
                   constant_expressions[i],
                   "\n in parsed function object.\n",
                   expression->ErrorMsg());

      constant_values[i] = expression->Eval(nullptr);

      if (!fp.AddConstant(constant_names[i], constant_values[i]))
        mooseError("Invalid constant name in parsed function object");
    }

    // build variables string
    const auto variables = MooseUtils::join(variables_vec, ",");

    // parse
    fp.Parse(expression, variables);

    if (fp.Parse(expression, variables) >= 0)
      paramError("expression", "Invalid function: ", fp.ErrorMsg());

    // take derivatives
    for (const auto & d : getParam<std::vector<TensorInputBufferName>>("derivatives"))
      if (std::find(names.begin(), names.end(), d) != names.end())
      {
        if (fp.AutoDiff(d) != -1)
          paramError("expression", "Failed to take derivative w.r.t. `", d, "`.");
      }
      else
        paramError("derivatives",
                   "Derivative w.r.t `",
                   d,
                   "` was requested, but it is not listed in `inputs`.");

    if (getParam<bool>("enable_fpoptimizer"))
      fp.Optimize();

    fp.setupTensors();
  };

  if (_use_jit)
    setup(_jit);
  else
    setup(_no_jit);
}

void
ParsedCompute::computeBuffer()
{
  if (_use_jit)
    _u = _jit.Eval(_params);
  else
    _u = _no_jit.Eval(_params);
}
