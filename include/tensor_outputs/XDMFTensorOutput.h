/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOutput.h"
#include "pugixml.h"
#include <thread>

#ifdef LIBMESH_HAVE_HDF5
#include "hdf5.h"
#endif

/**
 * Postprocessor that operates on a buffer
 */
class XDMFTensorOutput : public TensorOutput
{
public:
  static InputParameters validParams();

  XDMFTensorOutput(const InputParameters & parameters);

  virtual void init() override;

protected:
  virtual void output() override;

  torch::Tensor extendTensor(torch::Tensor tensor);

  /// mesh dimension
  const unsigned int _dim;

  /// xml document references
  pugi::xml_document _doc;
  pugi::xml_node _tgrid;

  /// node grid is original buffer dimensions plus one
  std::vector<std::size_t> _nnode;
  std::string _node_grid;

  /// data dimensions (depends on choice of Cell or Node output)
  std::array<std::vector<std::size_t>, 2> _ndata;
  std::array<std::string, 2> _data_grid;
  /// file name base
  std::string _file_base;

  /// outputted frame
  std::size_t _frame;

  /// whether the tensor uses Cell or Node output
  std::map<std::string, bool> _is_cell_data;

#ifdef LIBMESH_HAVE_HDF5
  const bool _enable_hdf5;

  /// File created
  bool _hdf5_file_created;
#endif
};
