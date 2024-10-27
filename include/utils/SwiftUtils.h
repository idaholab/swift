//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include <torch/torch.h>

#define pti(tensor) MooseTensor::printTensorInfo(#tensor, tensor)

#define pez(tensor) MooseTensor::printElementZero(#tensor, tensor)

namespace MooseTensor
{

/// Passkey pattern key template
template <typename T>
class Key
{
  friend T;
  Key() {}
  Key(Key const &) {}
};

void printTensorInfo(const torch::Tensor & x);
void printTensorInfo(const std::string & name, const torch::Tensor & x);

void printElementZero(const torch::Tensor & tensor);
void printElementZero(const std::string & name, const torch::Tensor & tensor);

const torch::TensorOptions floatTensorOptions();
const torch::TensorOptions complexFloatTensorOptions();
const torch::TensorOptions intTensorOptions();

} // namespace MooseTensor

