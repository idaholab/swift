//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#ifdef NEML2_ENABLED

#include <torch/torch.h>

namespace MooseTensor
{

void printTensorInfo(const torch::Tensor & x);

const torch::TensorOptions floatTensorOptions();
const torch::TensorOptions intTensorOptions();

} // namespace MooseTensor

#endif
