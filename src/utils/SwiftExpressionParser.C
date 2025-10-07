/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "SwiftExpressionParser.h"
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <stdexcept>
#include <sstream>
#include <cmath>

namespace SwiftExpressionParser
{

// Helper to check if an expression is a constant
static std::shared_ptr<Constant> asConstant(const ExprPtr & expr)
{
  return std::dynamic_pointer_cast<Constant>(expr);
}

//=============================================================================
// BinaryOp implementation
//=============================================================================

const char * BinaryOp::opString(Op op)
{
  switch (op)
  {
    case Op::Add: return "+";
    case Op::Sub: return "-";
    case Op::Mul: return "*";
    case Op::Div: return "/";
    case Op::Pow: return "^";
  }
  return "?";
}

ExprPtr BinaryOp::simplify() const
{
  auto left = _left->simplify();
  auto right = _right->simplify();

  auto lc = asConstant(left);
  auto rc = asConstant(right);

  // Both constants - fold
  if (lc && rc)
  {
    double result = 0.0;
    switch (_op)
    {
      case Op::Add: result = lc->value() + rc->value(); break;
      case Op::Sub: result = lc->value() - rc->value(); break;
      case Op::Mul: result = lc->value() * rc->value(); break;
      case Op::Div: result = lc->value() / rc->value(); break;
      case Op::Pow: result = std::pow(lc->value(), rc->value()); break;
    }
    return std::make_shared<Constant>(result);
  }

  // Algebraic simplifications
  switch (_op)
  {
    case Op::Add:
      if (lc && lc->value() == 0.0) return right;
      if (rc && rc->value() == 0.0) return left;
      break;

    case Op::Sub:
      if (rc && rc->value() == 0.0) return left;
      if (lc && lc->value() == 0.0)
        return std::make_shared<UnaryOp>(UnaryOp::Op::Neg, right)->simplify();
      break;

    case Op::Mul:
      if (lc && lc->value() == 0.0) return std::make_shared<Constant>(0.0);
      if (rc && rc->value() == 0.0) return std::make_shared<Constant>(0.0);
      if (lc && lc->value() == 1.0) return right;
      if (rc && rc->value() == 1.0) return left;
      if (lc && lc->value() == -1.0)
        return std::make_shared<UnaryOp>(UnaryOp::Op::Neg, right)->simplify();
      if (rc && rc->value() == -1.0)
        return std::make_shared<UnaryOp>(UnaryOp::Op::Neg, left)->simplify();
      break;

    case Op::Div:
      if (lc && lc->value() == 0.0) return std::make_shared<Constant>(0.0);
      if (rc && rc->value() == 1.0) return left;
      break;

    case Op::Pow:
      if (rc && rc->value() == 0.0) return std::make_shared<Constant>(1.0);
      if (rc && rc->value() == 1.0) return left;
      if (lc && lc->value() == 1.0) return std::make_shared<Constant>(1.0);
      break;
  }

  return std::make_shared<BinaryOp>(_op, left, right);
}

ExprPtr BinaryOp::differentiate(const std::string & var) const
{
  auto dl = _left->differentiate(var);
  auto dr = _right->differentiate(var);

  switch (_op)
  {
    case Op::Add:
      return std::make_shared<BinaryOp>(Op::Add, dl, dr);

    case Op::Sub:
      return std::make_shared<BinaryOp>(Op::Sub, dl, dr);

    case Op::Mul:
      // (f*g)' = f'*g + f*g'
      return std::make_shared<BinaryOp>(
          Op::Add,
          std::make_shared<BinaryOp>(Op::Mul, dl, _right),
          std::make_shared<BinaryOp>(Op::Mul, _left, dr));

    case Op::Div:
      // (f/g)' = (f'*g - f*g') / g^2
      return std::make_shared<BinaryOp>(
          Op::Div,
          std::make_shared<BinaryOp>(
              Op::Sub,
              std::make_shared<BinaryOp>(Op::Mul, dl, _right),
              std::make_shared<BinaryOp>(Op::Mul, _left, dr)),
          std::make_shared<BinaryOp>(Op::Pow, _right, std::make_shared<Constant>(2.0)));

    case Op::Pow:
      // For f^g where g is constant: (f^g)' = g * f^(g-1) * f'
      if (auto rc = asConstant(_right))
      {
        return std::make_shared<BinaryOp>(
            Op::Mul,
            std::make_shared<BinaryOp>(
                Op::Mul,
                _right,
                std::make_shared<BinaryOp>(
                    Op::Pow,
                    _left,
                    std::make_shared<Constant>(rc->value() - 1.0))),
            dl);
      }
      // General case: (f^g)' = f^g * (g' * ln(f) + g * f'/f)
      return std::make_shared<BinaryOp>(
          Op::Mul,
          std::make_shared<BinaryOp>(Op::Pow, _left, _right),
          std::make_shared<BinaryOp>(
              Op::Add,
              std::make_shared<BinaryOp>(
                  Op::Mul,
                  dr,
                  std::make_shared<FunctionCall>("log", std::vector<ExprPtr>{_left})),
              std::make_shared<BinaryOp>(
                  Op::Mul,
                  _right,
                  std::make_shared<BinaryOp>(Op::Div, dl, _left))));
  }
  throw std::runtime_error("Invalid binary operator");
}

torch::jit::Value * BinaryOp::buildGraph(
    torch::jit::Graph & graph,
    std::unordered_map<std::string, torch::jit::Value *> & vars) const
{
  auto left = _left->buildGraph(graph, vars);
  auto right = _right->buildGraph(graph, vars);

  switch (_op)
  {
    case Op::Add: return graph.insert(torch::jit::aten::add, {left, right});
    case Op::Sub: return graph.insert(torch::jit::aten::sub, {left, right});
    case Op::Mul: return graph.insert(torch::jit::aten::mul, {left, right});
    case Op::Div: return graph.insert(torch::jit::aten::div, {left, right});
    case Op::Pow: return graph.insert(torch::jit::aten::pow, {left, right});
  }
  throw std::runtime_error("Invalid binary operator");
}

std::string BinaryOp::toString() const
{
  return "(" + _left->toString() + " " + opString(_op) + " " + _right->toString() + ")";
}

//=============================================================================
// UnaryOp implementation
//=============================================================================

ExprPtr UnaryOp::simplify() const
{
  auto operand = _operand->simplify();

  if (auto c = asConstant(operand))
  {
    switch (_op)
    {
      case Op::Neg: return std::make_shared<Constant>(-c->value());
      case Op::Not: return std::make_shared<Constant>(c->value() == 0.0 ? 1.0 : 0.0);
    }
  }

  return std::make_shared<UnaryOp>(_op, operand);
}

ExprPtr UnaryOp::differentiate(const std::string & var) const
{
  switch (_op)
  {
    case Op::Neg:
      return std::make_shared<UnaryOp>(Op::Neg, _operand->differentiate(var));
    case Op::Not:
      return std::make_shared<Constant>(0.0); // Derivative of logical not is zero
  }
  throw std::runtime_error("Invalid unary operator");
}

torch::jit::Value * UnaryOp::buildGraph(
    torch::jit::Graph & graph,
    std::unordered_map<std::string, torch::jit::Value *> & vars) const
{
  auto operand = _operand->buildGraph(graph, vars);

  switch (_op)
  {
    case Op::Neg:
      return graph.insert(torch::jit::aten::neg, {operand});
    case Op::Not:
      return graph.insert(torch::jit::aten::logical_not, {operand});
  }
  throw std::runtime_error("Invalid unary operator");
}

std::string UnaryOp::toString() const
{
  switch (_op)
  {
    case Op::Neg: return "(-" + _operand->toString() + ")";
    case Op::Not: return "(!" + _operand->toString() + ")";
  }
  return "?";
}

//=============================================================================
// Comparison implementation
//=============================================================================

ExprPtr Comparison::simplify() const
{
  auto left = _left->simplify();
  auto right = _right->simplify();

  auto lc = asConstant(left);
  auto rc = asConstant(right);

  if (lc && rc)
  {
    bool result = false;
    switch (_op)
    {
      case Op::Lt: result = lc->value() < rc->value(); break;
      case Op::Gt: result = lc->value() > rc->value(); break;
      case Op::Le: result = lc->value() <= rc->value(); break;
      case Op::Ge: result = lc->value() >= rc->value(); break;
      case Op::Eq: result = lc->value() == rc->value(); break;
      case Op::Ne: result = lc->value() != rc->value(); break;
    }
    return std::make_shared<Constant>(result ? 1.0 : 0.0);
  }

  return std::make_shared<Comparison>(_op, left, right);
}

ExprPtr Comparison::differentiate(const std::string &) const
{
  // Derivative of comparison is zero (not differentiable in classical sense)
  return std::make_shared<Constant>(0.0);
}

torch::jit::Value * Comparison::buildGraph(
    torch::jit::Graph & graph,
    std::unordered_map<std::string, torch::jit::Value *> & vars) const
{
  auto left = _left->buildGraph(graph, vars);
  auto right = _right->buildGraph(graph, vars);

  switch (_op)
  {
    case Op::Lt: return graph.insert(torch::jit::aten::lt, {left, right});
    case Op::Gt: return graph.insert(torch::jit::aten::gt, {left, right});
    case Op::Le: return graph.insert(torch::jit::aten::le, {left, right});
    case Op::Ge: return graph.insert(torch::jit::aten::ge, {left, right});
    case Op::Eq: return graph.insert(torch::jit::aten::eq, {left, right});
    case Op::Ne: return graph.insert(torch::jit::aten::ne, {left, right});
  }
  throw std::runtime_error("Invalid comparison operator");
}

std::string Comparison::toString() const
{
  const char * op = "?";
  switch (_op)
  {
    case Op::Lt: op = "<"; break;
    case Op::Gt: op = ">"; break;
    case Op::Le: op = "<="; break;
    case Op::Ge: op = ">="; break;
    case Op::Eq: op = "=="; break;
    case Op::Ne: op = "!="; break;
  }
  return "(" + _left->toString() + " " + op + " " + _right->toString() + ")";
}

//=============================================================================
// LogicalOp implementation
//=============================================================================

ExprPtr LogicalOp::simplify() const
{
  auto left = _left->simplify();
  auto right = _right->simplify();

  auto lc = asConstant(left);
  auto rc = asConstant(right);

  if (lc && rc)
  {
    bool lv = lc->value() != 0.0;
    bool rv = rc->value() != 0.0;
    bool result = false;
    switch (_op)
    {
      case Op::And: result = lv && rv; break;
      case Op::Or: result = lv || rv; break;
    }
    return std::make_shared<Constant>(result ? 1.0 : 0.0);
  }

  // Algebraic simplifications
  switch (_op)
  {
    case Op::And:
      if (lc && lc->value() == 0.0) return std::make_shared<Constant>(0.0);
      if (rc && rc->value() == 0.0) return std::make_shared<Constant>(0.0);
      break;
    case Op::Or:
      if (lc && lc->value() != 0.0) return std::make_shared<Constant>(1.0);
      if (rc && rc->value() != 0.0) return std::make_shared<Constant>(1.0);
      break;
  }

  return std::make_shared<LogicalOp>(_op, left, right);
}

ExprPtr LogicalOp::differentiate(const std::string &) const
{
  // Derivative of logical operations is zero
  return std::make_shared<Constant>(0.0);
}

torch::jit::Value * LogicalOp::buildGraph(
    torch::jit::Graph & graph,
    std::unordered_map<std::string, torch::jit::Value *> & vars) const
{
  auto left = _left->buildGraph(graph, vars);
  auto right = _right->buildGraph(graph, vars);

  switch (_op)
  {
    case Op::And: return graph.insert(torch::jit::aten::logical_and, {left, right});
    case Op::Or: return graph.insert(torch::jit::aten::logical_or, {left, right});
  }
  throw std::runtime_error("Invalid logical operator");
}

std::string LogicalOp::toString() const
{
  const char * op = (_op == Op::And) ? "&" : "|";
  return "(" + _left->toString() + " " + op + " " + _right->toString() + ")";
}

//=============================================================================
// FunctionCall implementation
//=============================================================================

ExprPtr FunctionCall::simplify() const
{
  // Simplify all arguments
  std::vector<ExprPtr> simplified_args;
  bool all_const = true;
  std::vector<double> const_values;

  for (const auto & arg : _args)
  {
    auto s = arg->simplify();
    simplified_args.push_back(s);
    if (auto c = asConstant(s))
      const_values.push_back(c->value());
    else
      all_const = false;
  }

  // If all arguments are constants, fold the function
  if (all_const)
  {
    double result = 0.0;
    if (_name == "sin" && const_values.size() == 1)
      result = std::sin(const_values[0]);
    else if (_name == "cos" && const_values.size() == 1)
      result = std::cos(const_values[0]);
    else if (_name == "tan" && const_values.size() == 1)
      result = std::tan(const_values[0]);
    else if (_name == "sinh" && const_values.size() == 1)
      result = std::sinh(const_values[0]);
    else if (_name == "cosh" && const_values.size() == 1)
      result = std::cosh(const_values[0]);
    else if (_name == "tanh" && const_values.size() == 1)
      result = std::tanh(const_values[0]);
    else if (_name == "asin" && const_values.size() == 1)
      result = std::asin(const_values[0]);
    else if (_name == "acos" && const_values.size() == 1)
      result = std::acos(const_values[0]);
    else if (_name == "atan" && const_values.size() == 1)
      result = std::atan(const_values[0]);
    else if (_name == "asinh" && const_values.size() == 1)
      result = std::asinh(const_values[0]);
    else if (_name == "acosh" && const_values.size() == 1)
      result = std::acosh(const_values[0]);
    else if (_name == "atanh" && const_values.size() == 1)
      result = std::atanh(const_values[0]);
    else if (_name == "exp" && const_values.size() == 1)
      result = std::exp(const_values[0]);
    else if (_name == "log" && const_values.size() == 1)
      result = std::log(const_values[0]);
    else if (_name == "log10" && const_values.size() == 1)
      result = std::log10(const_values[0]);
    else if (_name == "log2" && const_values.size() == 1)
      result = std::log2(const_values[0]);
    else if (_name == "sqrt" && const_values.size() == 1)
      result = std::sqrt(const_values[0]);
    else if (_name == "abs" && const_values.size() == 1)
      result = std::abs(const_values[0]);
    else if (_name == "ceil" && const_values.size() == 1)
      result = std::ceil(const_values[0]);
    else if (_name == "floor" && const_values.size() == 1)
      result = std::floor(const_values[0]);
    else if (_name == "round" && const_values.size() == 1)
      result = std::round(const_values[0]);
    else if (_name == "trunc" && const_values.size() == 1)
      result = std::trunc(const_values[0]);
    else if (_name == "min" && const_values.size() == 2)
      result = std::min(const_values[0], const_values[1]);
    else if (_name == "max" && const_values.size() == 2)
      result = std::max(const_values[0], const_values[1]);
    else if (_name == "atan2" && const_values.size() == 2)
      result = std::atan2(const_values[0], const_values[1]);
    else if (_name == "hypot" && const_values.size() == 2)
      result = std::hypot(const_values[0], const_values[1]);
    else if (_name == "pow" && const_values.size() == 2)
      result = std::pow(const_values[0], const_values[1]);
    else if (_name == "if" && const_values.size() == 3)
      result = const_values[0] != 0.0 ? const_values[1] : const_values[2];
    else
      return std::make_shared<FunctionCall>(_name, simplified_args);

    return std::make_shared<Constant>(result);
  }

  return std::make_shared<FunctionCall>(_name, simplified_args);
}

ExprPtr FunctionCall::differentiate(const std::string & var) const
{
  if (_args.empty())
    return std::make_shared<Constant>(0.0);

  auto arg = _args[0];
  auto darg = arg->differentiate(var);

  // Chain rule for single-argument functions
  if (_name == "sin")
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Mul,
        std::make_shared<FunctionCall>("cos", std::vector<ExprPtr>{arg}),
        darg);
  else if (_name == "cos")
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Mul,
        std::make_shared<UnaryOp>(
            UnaryOp::Op::Neg,
            std::make_shared<FunctionCall>("sin", std::vector<ExprPtr>{arg})),
        darg);
  else if (_name == "tan")
  {
    auto cos_arg = std::make_shared<FunctionCall>("cos", std::vector<ExprPtr>{arg});
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Div,
        darg,
        std::make_shared<BinaryOp>(BinaryOp::Op::Mul, cos_arg, cos_arg));
  }
  else if (_name == "sinh")
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Mul,
        std::make_shared<FunctionCall>("cosh", std::vector<ExprPtr>{arg}),
        darg);
  else if (_name == "cosh")
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Mul,
        std::make_shared<FunctionCall>("sinh", std::vector<ExprPtr>{arg}),
        darg);
  else if (_name == "tanh")
  {
    auto cosh_arg = std::make_shared<FunctionCall>("cosh", std::vector<ExprPtr>{arg});
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Div,
        darg,
        std::make_shared<BinaryOp>(BinaryOp::Op::Mul, cosh_arg, cosh_arg));
  }
  else if (_name == "exp")
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Mul,
        std::make_shared<FunctionCall>("exp", std::vector<ExprPtr>{arg}),
        darg);
  else if (_name == "log")
    return std::make_shared<BinaryOp>(BinaryOp::Op::Div, darg, arg);
  else if (_name == "log10")
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Div,
        darg,
        std::make_shared<BinaryOp>(
            BinaryOp::Op::Mul,
            arg,
            std::make_shared<FunctionCall>("log", std::vector<ExprPtr>{std::make_shared<Constant>(10.0)})));
  else if (_name == "log2")
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Div,
        darg,
        std::make_shared<BinaryOp>(
            BinaryOp::Op::Mul,
            arg,
            std::make_shared<FunctionCall>("log", std::vector<ExprPtr>{std::make_shared<Constant>(2.0)})));
  else if (_name == "sqrt")
    return std::make_shared<BinaryOp>(
        BinaryOp::Op::Div,
        darg,
        std::make_shared<BinaryOp>(
            BinaryOp::Op::Mul,
            std::make_shared<Constant>(2.0),
            std::make_shared<FunctionCall>("sqrt", std::vector<ExprPtr>{arg})));
  else if (_name == "if" && _args.size() == 3)
  {
    // Derivative of if(c, t, f) w.r.t. var is if(c, dt/dvar, df/dvar)
    return std::make_shared<FunctionCall>(
        "if",
        std::vector<ExprPtr>{_args[0],
                             _args[1]->differentiate(var),
                             _args[2]->differentiate(var)});
  }

  // For other functions, return zero (not differentiable or not implemented)
  return std::make_shared<Constant>(0.0);
}

torch::jit::Value * FunctionCall::buildGraph(
    torch::jit::Graph & graph,
    std::unordered_map<std::string, torch::jit::Value *> & vars) const
{
  std::vector<torch::jit::Value *> arg_vals;
  for (const auto & arg : _args)
    arg_vals.push_back(arg->buildGraph(graph, vars));

  // Map function names to torch operations
  if (_name == "sin" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::sin, {arg_vals[0]});
  else if (_name == "cos" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::cos, {arg_vals[0]});
  else if (_name == "tan" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::tan, {arg_vals[0]});
  else if (_name == "sinh" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::sinh, {arg_vals[0]});
  else if (_name == "cosh" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::cosh, {arg_vals[0]});
  else if (_name == "tanh" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::tanh, {arg_vals[0]});
  else if (_name == "asin" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::asin, {arg_vals[0]});
  else if (_name == "acos" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::acos, {arg_vals[0]});
  else if (_name == "atan" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::atan, {arg_vals[0]});
  else if (_name == "asinh" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::asinh, {arg_vals[0]});
  else if (_name == "acosh" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::acosh, {arg_vals[0]});
  else if (_name == "atanh" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::atanh, {arg_vals[0]});
  else if (_name == "exp" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::exp, {arg_vals[0]});
  else if (_name == "exp2" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::exp2, {arg_vals[0]});
  else if (_name == "log" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::log, {arg_vals[0]});
  else if (_name == "log10" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::log10, {arg_vals[0]});
  else if (_name == "log2" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::log2, {arg_vals[0]});
  else if (_name == "sqrt" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::sqrt, {arg_vals[0]});
  else if (_name == "rsqrt" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::rsqrt, {arg_vals[0]});
  else if (_name == "abs" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::abs, {arg_vals[0]});
  else if (_name == "ceil" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::ceil, {arg_vals[0]});
  else if (_name == "floor" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::floor, {arg_vals[0]});
  else if (_name == "round" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::round, {arg_vals[0]});
  else if (_name == "trunc" && arg_vals.size() == 1)
    return graph.insert(torch::jit::aten::trunc, {arg_vals[0]});
  else if (_name == "min" && arg_vals.size() == 2)
    return graph.insert(torch::jit::aten::minimum, {arg_vals[0], arg_vals[1]});
  else if (_name == "max" && arg_vals.size() == 2)
    return graph.insert(torch::jit::aten::maximum, {arg_vals[0], arg_vals[1]});
  else if (_name == "atan2" && arg_vals.size() == 2)
    return graph.insert(torch::jit::aten::atan2, {arg_vals[0], arg_vals[1]});
  else if (_name == "hypot" && arg_vals.size() == 2)
    return graph.insert(torch::jit::aten::hypot, {arg_vals[0], arg_vals[1]});
  else if (_name == "pow" && arg_vals.size() == 2)
    return graph.insert(torch::jit::aten::pow, {arg_vals[0], arg_vals[1]});
  else if (_name == "fmod" && arg_vals.size() == 2)
    return graph.insert(torch::jit::aten::fmod, {arg_vals[0], arg_vals[1]});
  else if (_name == "if" && arg_vals.size() == 3)
    return graph.insert(torch::jit::aten::where, {arg_vals[0], arg_vals[1], arg_vals[2]});
  else if (_name == "FFT" && arg_vals.size() == 1)
  {
    // Handle FFT specially - need to determine dimensions
    auto input = arg_vals[0];
    // For now, we'll use rfft (1D), but this should be determined based on input dimensions
    return graph.insert(torch::jit::aten::fft_rfft, {input});
  }
  else if (_name == "iFFT" && arg_vals.size() == 1)
  {
    auto input = arg_vals[0];
    return graph.insert(torch::jit::aten::fft_irfft, {input});
  }

  throw std::runtime_error("Unknown or unsupported function: " + _name);
}

std::string FunctionCall::toString() const
{
  std::string result = _name + "(";
  for (size_t i = 0; i < _args.size(); ++i)
  {
    if (i > 0) result += ", ";
    result += _args[i]->toString();
  }
  result += ")";
  return result;
}

//=============================================================================
// Parser implementation
//=============================================================================

Parser::Parser()
{
  if (!_parser.load_grammar(grammar))
    throw std::runtime_error("Failed to load expression grammar");

  // Set up semantic actions to build AST
  _parser["NUMBER"] = [](const peg::SemanticValues & sv) -> ExprPtr {
    return std::make_shared<Constant>(std::stod(sv.token_to_string()));
  };

  _parser["VARIABLE"] = [](const peg::SemanticValues & sv) -> ExprPtr {
    std::string name = sv.token_to_string();
    // Trim whitespace
    name.erase(0, name.find_first_not_of(" \t\r\n"));
    name.erase(name.find_last_not_of(" \t\r\n") + 1);
    return std::make_shared<Variable>(name);
  };

  _parser["IDENTIFIER"] = [](const peg::SemanticValues & sv) {
    std::string name = sv.token_to_string();
    // Trim whitespace
    name.erase(0, name.find_first_not_of(" \t\r\n"));
    name.erase(name.find_last_not_of(" \t\r\n") + 1);
    return name;
  };

  _parser["PRIMARY"] = [](const peg::SemanticValues & sv) -> ExprPtr {
    return std::any_cast<ExprPtr>(sv[0]);
  };

  _parser["POWER"] = [](const peg::SemanticValues & sv) -> ExprPtr {
    ExprPtr result = std::any_cast<ExprPtr>(sv[0]);
    if (sv.size() > 1)
      result = std::make_shared<BinaryOp>(BinaryOp::Op::Pow, result, std::any_cast<ExprPtr>(sv[1]));
    return result;
  };

  _parser["UNARY_OP"] = [](const peg::SemanticValues & sv) {
    return sv.token_to_string();
  };

  _parser["UNARY"] = [](const peg::SemanticValues & sv) -> ExprPtr {
    if (sv.size() == 1)
      return std::any_cast<ExprPtr>(sv[0]);

    std::string op = std::any_cast<std::string>(sv[0]);
    ExprPtr operand = std::any_cast<ExprPtr>(sv[1]);

    if (op == "-")
      return std::make_shared<UnaryOp>(UnaryOp::Op::Neg, operand);
    else if (op == "!")
      return std::make_shared<UnaryOp>(UnaryOp::Op::Not, operand);

    throw std::runtime_error("Unknown unary operator: " + op);
  };

  _parser["MUL_OP"] = [](const peg::SemanticValues & sv) {
    return sv.token_to_string();
  };

  _parser["MULTITIVE"] = [](const peg::SemanticValues & sv) -> ExprPtr {
    ExprPtr result = std::any_cast<ExprPtr>(sv[0]);
    for (size_t i = 1; i < sv.size(); i += 2)
    {
      std::string op = std::any_cast<std::string>(sv[i]);
      ExprPtr right = std::any_cast<ExprPtr>(sv[i + 1]);

      if (op == "*")
        result = std::make_shared<BinaryOp>(BinaryOp::Op::Mul, result, right);
      else if (op == "/")
        result = std::make_shared<BinaryOp>(BinaryOp::Op::Div, result, right);
    }
    return result;
  };

  _parser["ADD_OP"] = [](const peg::SemanticValues & sv) {
    return sv.token_to_string();
  };

  _parser["ADDITIVE"] = [](const peg::SemanticValues & sv) -> ExprPtr {
    ExprPtr result = std::any_cast<ExprPtr>(sv[0]);
    for (size_t i = 1; i < sv.size(); i += 2)
    {
      std::string op = std::any_cast<std::string>(sv[i]);
      ExprPtr right = std::any_cast<ExprPtr>(sv[i + 1]);

      if (op == "+")
        result = std::make_shared<BinaryOp>(BinaryOp::Op::Add, result, right);
      else if (op == "-")
        result = std::make_shared<BinaryOp>(BinaryOp::Op::Sub, result, right);
    }
    return result;
  };

  _parser["COMP_OP"] = [](const peg::SemanticValues & sv) {
    return sv.token_to_string();
  };

  _parser["COMPARISON"] = [](const peg::SemanticValues & sv) -> ExprPtr {
    if (sv.size() == 1)
      return std::any_cast<ExprPtr>(sv[0]);

    ExprPtr left = std::any_cast<ExprPtr>(sv[0]);
    std::string op = std::any_cast<std::string>(sv[1]);
    ExprPtr right = std::any_cast<ExprPtr>(sv[2]);

    Comparison::Op cmp_op;
    if (op == "<")
      cmp_op = Comparison::Op::Lt;
    else if (op == ">")
      cmp_op = Comparison::Op::Gt;
    else if (op == "<=")
      cmp_op = Comparison::Op::Le;
    else if (op == ">=")
      cmp_op = Comparison::Op::Ge;
    else if (op == "==")
      cmp_op = Comparison::Op::Eq;
    else if (op == "!=")
      cmp_op = Comparison::Op::Ne;
    else
      throw std::runtime_error("Unknown comparison operator: " + op);

    return std::make_shared<Comparison>(cmp_op, left, right);
  };

  _parser["LOGICAL_OP"] = [](const peg::SemanticValues & sv) {
    return sv.token_to_string();
  };

  _parser["LOGICAL"] = [](const peg::SemanticValues & sv) -> ExprPtr {
    ExprPtr result = std::any_cast<ExprPtr>(sv[0]);
    for (size_t i = 1; i < sv.size(); i += 2)
    {
      std::string op = std::any_cast<std::string>(sv[i]);
      ExprPtr right = std::any_cast<ExprPtr>(sv[i + 1]);

      if (op == "&")
        result = std::make_shared<LogicalOp>(LogicalOp::Op::And, result, right);
      else if (op == "|")
        result = std::make_shared<LogicalOp>(LogicalOp::Op::Or, result, right);
    }
    return result;
  };

  _parser["ARGS"] = [](const peg::SemanticValues & sv) {
    std::vector<ExprPtr> args;
    for (const auto & v : sv)
      args.push_back(std::any_cast<ExprPtr>(v));
    return args;
  };

  _parser["FUNCTION"] = [](const peg::SemanticValues & sv) -> ExprPtr {
    std::string name = std::any_cast<std::string>(sv[0]);
    std::vector<ExprPtr> args;
    if (sv.size() > 1)
      args = std::any_cast<std::vector<ExprPtr>>(sv[1]);
    return std::make_shared<FunctionCall>(name, args);
  };

  _parser["EXPRESSION"] = [](const peg::SemanticValues & sv) -> ExprPtr {
    return std::any_cast<ExprPtr>(sv[0]);
  };
}

ExprPtr Parser::parse(const std::string & expr)
{
  ExprPtr result;
  _error.clear();

  _parser.set_logger([this](size_t line, size_t col, const std::string & msg, const std::string &) {
    _error = "Line " + std::to_string(line) + ":" + std::to_string(col) + ": " + msg;
  });

  if (!_parser.parse(expr, result))
  {
    if (_error.empty())
      _error = "Failed to parse expression";
    return nullptr;
  }

  return result;
}

} // namespace SwiftExpressionParser
