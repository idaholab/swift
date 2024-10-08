
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "DomainInterface.h"
#include "DomainAction.h"
#include "ActionWarehouse.h"
#include "MooseBase.h"
#include "MooseApp.h"

DomainInterface::DomainInterface(MooseBase * moose_base) : _domain(getDomain(moose_base)) {}

const DomainAction &
DomainInterface::getDomain(MooseBase * moose_base)
{
  auto actions = moose_base->getMooseApp().actionWarehouse().getActions<DomainAction>();
  if (actions.size() != 1)
    mooseError("No [Domain] block found in input.");
  return *actions[0];
}
