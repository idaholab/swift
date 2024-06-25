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
  FFTRawXDMFOut(const FFTProblem & fft_problem);

protected:
  virtual void output() override;

  /// xml document references
  pugi::xml_document _xmdf;
  pugi::xml_node _tgrid;
  pugi::xml_attribute _tsize;
  pugi::xml_node _tdata;

  std::string _ngrid;

  /// Timesteps
  std::vector<Real> _times;

  /// outputted frame
  std::size_t _frame;
};
