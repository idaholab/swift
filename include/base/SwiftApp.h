/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

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
