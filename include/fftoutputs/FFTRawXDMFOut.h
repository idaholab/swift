//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "FFTOutput.h"
#include "pugixml.h"
#include <thread>

/**
 * Postprocessor that operates on a buffer
 */
class FFTRawXDMFOut : public FFTOutput
{
public:
  FFTRawXDMFOut(const InputParameters & parameters);

  virtual void init() override;

protected:
  virtual void output() override;

  /// xml document references
  pugi::xml_document _doc;
  pugi::xml_node _tgrid;

  std::string _ngrid;

  /// outputted frame
  std::size_t _frame;
};
