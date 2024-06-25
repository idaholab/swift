//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "FFTRawXDMFOut.h"
#include "FFTProblem.h"
#include "Conversion.h"

FFTRawXDMFOut::FFTRawXDMFOut(const FFTProblem & fft_problem)
  : FFTOutput(fft_problem), _domain(_xmdf.append_child("Domain"))
{
  // get mesh metadata
  auto dim = _fft_problem.getDim();
  auto sdim = Moose::stringify(dim);
  std::vector<unsigned int> ngrid;
  std::vector<Real> dgrid;
  std::vector<Real> origin;
  for (const auto i : make_range(dim))
  {
    ngrid.push_back(_fft_problem.getGridSize()[i]);
    dgrid.push_back(_fft_problem.getGridSpacing()[i]);
    origin.push_back(0.0);
  }

  //
  // setup XDMF skeleton
  //

  // Topology
  auto topology = _domain.append_child("Topology");
  topology.append_attribute("name") = "Topo1";
  topology.append_attribute("TopologyType").set_value((sdim + "DCoRectMesh").c_str());
  topology.append_attribute("Dimensions").set_value(Moose::stringify(ngrid, " ").c_str());

  // Geometry
  auto geometry = _domain.append_child("Geometry");
  topology.append_attribute("name").set_value("Geom1");
  std::string type = "ORIGIN_";
  for (const auto i : make_range(dim))
    type += "D" + std::string('X' + char(i), 1);
  topology.append_attribute("Type").set_value(type.c_str());

  // - Origin
  {
    auto data = geometry.append_child("DataItem");
    data.append_attribute("Format").set_value("XML");
    data.append_attribute("Dimension").set_value(sdim.c_str());
    data.set_value(Moose::stringify(origin, " ").c_str());
  }

  // - Grid spacing

  _xmdf.save_file("save_file_output.xml");
}

void
FFTRawXDMFOut::output()
{
  // write raw buffer

  // update timesteps

  // add grids for new timestep

  // write XDMF file
}
