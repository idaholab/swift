//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "NEML2Utils.h"
#include "neml2/tensors/Scalar.h"

#include "libmesh/fparser_ad.hh"

class ParsedTensor : public FunctionParserAD
{
public:
  ParsedTensor() : FunctionParserAD(), _data(*getParserData()) {}

  void setupTensors();

  /// overload for torch tensors
  neml2::Scalar Eval(const neml2::Scalar * params);

protected:
  // we'll need a stack pool to make this thread safe
  std::vector<neml2::Scalar> s;

  // immediate values converted to tensors
  std::vector<neml2::Scalar> tensor_immed;

  const Data & _data;
};
