/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "ExpressionParser.h"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>

/**
 * ParsedTensor - mathematical expression parser for torch tensors
 *
 * This class uses a PEG-based parser to parse mathematical expressions,
 * build an AST, perform symbolic differentiation and algebraic simplification,
 * and generate optimized code for evaluation.
 */
class ParsedTensor
{
public:
  ParsedTensor();

  /// Parse an expression with given variable names
  bool parse(const std::string & expression, const std::vector<std::string> & variables);

  /// Take derivative with respect to a variable
  void differentiate(const std::string & var);

  /// Optimize the expression (constant folding, algebraic simplification)
  void optimize();

  /// Evaluate the expression with torch tensors
  torch::Tensor eval(const std::vector<const torch::Tensor *> & params);

  /// Get the error message from the last operation
  const std::string & errorMessage() const { return _error; }

  /// Get a string representation of the parsed expression
  std::string toString() const;

protected:
  ExprParser::Parser _parser;
  ExprParser::ExprPtr _ast;
  std::vector<std::string> _variables;
  std::string _error;
};
