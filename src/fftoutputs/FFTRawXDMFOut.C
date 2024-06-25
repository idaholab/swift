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

FFTRawXDMFOut::FFTRawXDMFOut(const FFTProblem & fft_problem) : FFTOutput(fft_problem), _frame(0)
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
  _ngrid = Moose::stringify(ngrid, " ");

  //
  // setup XDMF skeleton
  //

  // Domain
  auto domain = _xmdf.append_child("Domain");

  // - Topology
  auto topology = domain.append_child("Topology");
  topology.append_attribute("name") = "Topo1";
  topology.append_attribute("TopologyType") = (sdim + "DCoRectMesh").c_str();
  topology.append_attribute("Dimensions").set_value(_ngrid.c_str());

  // -  Geometry
  auto geometry = domain.append_child("Geometry");
  topology.append_attribute("name") = "Geom1";
  std::string type = "ORIGIN_";
  const char * dxyz[] = {"DX", "DY", "DZ"};
  for (const auto i : make_range(dim))
    type += dxyz[i];
  topology.append_attribute("Type") = type.c_str();

  // -- Origin
  {
    auto data = geometry.append_child("DataItem");
    data.append_attribute("Format").set_value("XML");
    data.append_attribute("Dimension") = sdim.c_str();
    data.append_child(pugi::node_pcdata).set_value(Moose::stringify(origin, " ").c_str());
  }

  // -- Grid spacing
  {
    auto data = geometry.append_child("DataItem");
    data.append_attribute("Format") = "XML";
    data.append_attribute("Dimension") = sdim.c_str();
    data.append_child(pugi::node_pcdata).set_value(Moose::stringify(dgrid, " ").c_str());
  }

  // - TimeSeries Grid
  _tgrid = domain.append_child("Grid");
  _tgrid.append_attribute("Name") = "TimeSeries";
  _tgrid.append_attribute("GridType") = "Collection";
  _tgrid.append_attribute("CollectionType") = "Temporal";

  // -- Times
  auto time = _tgrid.append_child("Time");
  time.append_attribute("TimeType") = "List";

  // --- Time data
  {
    auto data = time.append_child("DataItem");
    data.append_attribute("Format") = "XML";
    data.append_attribute("NumberType") = "Float";
    _tsize = data.append_attribute("Dimension");
    _tdata = data.append_child(pugi::node_pcdata);
  }

  output();
  output();
}

void
FFTRawXDMFOut::output()
{
  // update timesteps
  _times.push_back(_fft_problem.time());
  _tsize.set_value(_times.size());
  _tdata.set_value(Moose::stringify(_times, " ").c_str());

  // add grid for new timestep
  auto grid = _tgrid.append_child("Grid");
  grid.append_attribute("Name") = ("T" + Moose::stringify(_frame)).c_str();
  grid.append_attribute("GridType") = "Uniform";

  // add references
  grid.append_child("Topology").append_attribute("Reference") = "/Xdmf/Domain/Topology[1]";
  grid.append_child("Geometry").append_attribute("Reference") = "/Xdmf/Domain/Geometry[1]";

  // loop over buffers
  for (const auto i : make_range(3))
  {
    auto name = "Buf" + Moose::stringify(i);

    auto attr = grid.append_child("Attribute");
    attr.append_attribute("Name") = name.c_str();
    attr.append_attribute("Center") = "Node"; // or "Cell"?
    auto data = attr.append_child("DataItem");
    data.append_attribute("Format") = "Binary";
    data.append_attribute("DataType") = "Float";
    data.append_attribute("Precision") = "8"; // or 4
    data.append_attribute("Endian") = "Big";
    data.append_attribute("Dimensions") = _ngrid.c_str();

    // save file
    auto fname = name + "." + Moose::stringify(_frame) + ".bin";

    data.append_child(pugi::node_pcdata).set_value(fname.c_str());
  }

  // write XDMF file
  _xmdf.save_file("save_file_output.xml");

  // increment frame
  _frame++;
}
