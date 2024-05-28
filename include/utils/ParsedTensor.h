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
  void setupTensors();
  neml2::Scalar customEval(const neml2::Scalar * Vars);

protected:
  std::vector<neml2::Scalar> s;
  std::vector<neml2::Scalar> tensor_immed;
};
