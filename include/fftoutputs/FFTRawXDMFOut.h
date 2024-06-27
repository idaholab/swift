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

#ifdef LIBMESH_HAVE_HDF5
#include "hdf5.h"
#endif

/**
 * Postprocessor that operates on a buffer
 */
class FFTRawXDMFOut : public FFTOutput
{
public:
  static InputParameters validParams();

  FFTRawXDMFOut(const InputParameters & parameters);

  virtual void init() override;

protected:
  virtual void output() override;

  torch::Tensor prepareTensor(const torch::Tensor & in);

  /// mesh dimension
  const unsigned int _dim;

  /// xml document references
  pugi::xml_document _doc;
  pugi::xml_node _tgrid;

  /// node grid is original buffer dimensions plus one
  std::vector<std::size_t> _nnode;
  std::string _node_grid;

  /// data dimensions (depends on choice of Cell or Node output)
  std::vector<std::size_t> _ndata;
  std::string _data_grid;

  /// Cell or node data?
  const bool _cell_data;

  /// file name base
  std::string _file_base;

  /// outputted frame
  std::size_t _frame;

#ifdef LIBMESH_HAVE_HDF5
  const bool _enable_hdf5;
#endif
};
