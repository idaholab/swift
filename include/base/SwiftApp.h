//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "MooseApp.h"
#include "SwiftUtils.h"

namespace MooseTensor
{
std::string torchDevice();
}

class DomainAction;

class SwiftApp : public MooseApp
{
public:
  static InputParameters validParams();

  SwiftApp(InputParameters parameters);
  virtual ~SwiftApp();

  static void registerApps();
  static void registerAll(Factory & f, ActionFactory & af, Syntax & s);

  /// called from the ComputeDevice action
  void setTorchDevice(std::string device, const MooseTensor::Key<DomainAction> &);
};
