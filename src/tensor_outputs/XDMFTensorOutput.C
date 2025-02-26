/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "XDMFTensorOutput.h"
#include "TensorProblem.h"
#include "Conversion.h"

#include <filesystem>

#ifdef LIBMESH_HAVE_HDF5
namespace
{
void addDataToHDF5(hid_t file_id,
                   const std::string & dataset_name,
                   const char * data,
                   std::vector<std::size_t> & ndim,
                   hid_t type);
}
#endif

registerMooseObject("SwiftApp", XDMFTensorOutput);

InputParameters
XDMFTensorOutput::validParams()
{
  auto params = TensorOutput::validParams();
  params.addClassDescription("Output a tensor in XDMF format.");
#ifdef LIBMESH_HAVE_HDF5
  params.addParam<bool>("enable_hdf5", "Use HDF5 for binary data storage.");
#endif
  MultiMooseEnum outputMode("CELL NODE");
  params.addParam<MultiMooseEnum>("output_mode", outputMode, "Output as cell or node data");
  return params;
}

XDMFTensorOutput::XDMFTensorOutput(const InputParameters & parameters)
  : TensorOutput(parameters),
    _dim(_domain.getDim()),
    _file_base(_app.getOutputFileBase(true)),
    _frame(0)
#ifdef LIBMESH_HAVE_HDF5
    ,
    _enable_hdf5(getParam<bool>("enable_hdf5")),
    _hdf5_name(_file_base + ".h5")
#endif
{
  const auto & output_mode = getParam<MultiMooseEnum>("output_mode");
  const auto nbuffers = _out_buffers.size();

  if (output_mode.size() == 0)
    // default all to Cell
    for (const auto & pair : _out_buffers)
      _is_cell_data[pair.first] = true;
  else if (output_mode.size() != nbuffers)
    paramError(
        "output_mode", "Specify one output mode per buffer.", output_mode.size(), " != ", nbuffers);
  else
  {
    const auto & buffer_name = getParam<std::vector<TensorInputBufferName>>("buffer");
    for (const auto i : make_range(nbuffers))
      _is_cell_data[buffer_name[i]] = (output_mode[i] == "CELL");
  }

#ifdef LIBMESH_HAVE_HDF5
  // Check if the library is thread-safe
  hbool_t is_threadsafe;
  H5is_library_threadsafe(&is_threadsafe);
  if (!is_threadsafe)
  {
    for (const auto & output : _tensor_problem.getOutputs())
      if (output.get() != this && dynamic_cast<XDMFTensorOutput *>(output.get()))
        mooseError(
            "Using an hdf5 library that is not threadsafe and multiple XDMF output objects. "
            "Consolidate the XDMF outputs or build Swift with a thread safe build of libhdf5.");
    mooseWarning("Using an hdf5 library that is not threadsafe.");
  }
#endif
}

XDMFTensorOutput::~XDMFTensorOutput()
{
#ifdef LIBMESH_HAVE_HDF5
  if (_enable_hdf5)
    H5Fclose(_hdf5_file_id);
#endif
}

void
XDMFTensorOutput::init()
{
  // get mesh metadata
  auto sdim = Moose::stringify(_dim);
  std::vector<Real> origin;
  std::vector<Real> dgrid;
  for (const auto i : make_range(_dim))
  {
    _ndata[0].push_back(_domain.getGridSize()[i]);
    _ndata[1].push_back(_domain.getGridSize()[i] + 1);
    _nnode.push_back(_domain.getGridSize()[i] + 1);
    dgrid.push_back(_domain.getGridSpacing()(i));
    origin.push_back(_domain.getDomainMin()(i));
  }
  _data_grid[0] = Moose::stringify(_ndata[0], " ");
  _data_grid[1] = Moose::stringify(_ndata[1], " ");
  _node_grid = Moose::stringify(_nnode, " ");

  //
  // setup XDMF skeleton
  //

  // Top level xdmf block
  auto xdmf = _doc.append_child("Xdmf");
  xdmf.append_attribute("xmlns:xi") = "http://www.w3.org/2003/XInclude";
  xdmf.append_attribute("Version") = "2.2";

  // Domain
  auto domain = xdmf.append_child("Domain");

  // - Topology
  auto topology = domain.append_child("Topology");
  topology.append_attribute("TopologyType") = (sdim + "DCoRectMesh").c_str();
  topology.append_attribute("Dimensions").set_value(_node_grid.c_str());

  // -  Geometry
  auto geometry = domain.append_child("Geometry");
  std::string type = "ORIGIN_";
  const char * dxyz[] = {"DX", "DY", "DZ"};
  for (const auto i : make_range(_dim))
    type += dxyz[i];
  geometry.append_attribute("Type") = type.c_str();

  // -- Origin
  {
    auto data = geometry.append_child("DataItem");
    data.append_attribute("Format").set_value("XML");
    data.append_attribute("Dimensions") = sdim.c_str();
    data.append_child(pugi::node_pcdata).set_value(Moose::stringify(origin, " ").c_str());
  }

  // -- Grid spacing
  {
    auto data = geometry.append_child("DataItem");
    data.append_attribute("Format") = "XML";
    data.append_attribute("Dimensions") = sdim.c_str();
    data.append_child(pugi::node_pcdata).set_value(Moose::stringify(dgrid, " ").c_str());
  }

  // - TimeSeries Grid
  _tgrid = domain.append_child("Grid");
  _tgrid.append_attribute("Name") = "TimeSeries";
  _tgrid.append_attribute("GridType") = "Collection";
  _tgrid.append_attribute("CollectionType") = "Temporal";

  // write XDMF file
  _doc.save_file((_file_base + ".xmf").c_str());
#ifdef LIBMESH_HAVE_HDF5
  // delete HDF5 file
  if (_enable_hdf5)
  {
    std::filesystem::remove(_hdf5_name);
    // open new file
    _hdf5_file_id = H5Fcreate(_hdf5_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (_hdf5_file_id < 0)
    {
      H5Eprint(H5E_DEFAULT, stderr);
      mooseError("Error opening HDF5 file '", _hdf5_name, "'.");
    }
  }
#endif
}

void
XDMFTensorOutput::output()
{
  mooseInfoRepeated("Writing XDMF file '", _file_base, ".xmf' for output.");
  // add grid for new timestep
  auto grid = _tgrid.append_child("Grid");
  grid.append_attribute("Name") = ("T" + Moose::stringify(_frame)).c_str();
  grid.append_attribute("GridType") = "Uniform";

  // time
  auto time = grid.append_child("Time");
  time.append_attribute("Value") = _time;

  // add references
  grid.append_child("xi:include").append_attribute("xpointer") = "xpointer(//Xdmf/Domain/Topology)";
  grid.append_child("xi:include").append_attribute("xpointer") = "xpointer(//Xdmf/Domain/Geometry)";

  // loop over buffers
  for (const auto & [name, original_buffer] : _out_buffers)
  {
    const auto is_cell = _is_cell_data[name];
    auto attr = grid.append_child("Attribute");
    attr.append_attribute("Name") = name.c_str();
    attr.append_attribute("Center") = is_cell ? "Cell" : "Node";
    auto data = attr.append_child("DataItem");
    data.append_attribute("DataType") = "Float";
    data.append_attribute("Dimensions") = _data_grid[is_cell ? 0 : 1].c_str();

    auto buffer = is_cell ? *original_buffer : extendTensor(*original_buffer);
    // save file
    const auto setname = name + "." + Moose::stringify(_frame);
    char * raw_ptr = static_cast<char *>(buffer.data_ptr());
    std::size_t raw_size = buffer.numel();

#ifdef LIBMESH_HAVE_HDF5
    if (_enable_hdf5)
    {
      if (buffer.dtype() == torch::kFloat32)
        addDataToHDF5(_hdf5_file_id, setname, raw_ptr, _ndata[is_cell ? 0 : 1], H5T_NATIVE_FLOAT);
      else if (buffer.dtype() == torch::kFloat64)
        addDataToHDF5(_hdf5_file_id, setname, raw_ptr, _ndata[is_cell ? 0 : 1], H5T_NATIVE_DOUBLE);
      else
        mooseError("Unsupported output type");

      data.append_attribute("Format") = "HDF";
      const auto h5path = _hdf5_name + ":/" + setname;
      data.append_child(pugi::node_pcdata).set_value(h5path.c_str());
    }
    else
#endif
    {
      if (buffer.dtype() == torch::kFloat32)
      {
        data.append_attribute("Precision") = "4";
        raw_size *= 4;
      }
      else if (buffer.dtype() == torch::kFloat64)
      {
        data.append_attribute("Precision") = "8";
        raw_size *= 8;
      }
      else
        mooseError("Unsupported output type");

      const auto fname = _file_base + "." + setname + ".bin";
      auto file = std::fstream(fname.c_str(), std::ios::out | std::ios::binary);
      file.write(raw_ptr, raw_size);
      file.close();

      data.append_attribute("Format") = "Binary";
      data.append_attribute("Endian") = "Little";
      data.append_child(pugi::node_pcdata).set_value(fname.c_str());
    }
  }

  // write XDMF file
  _doc.save_file((_file_base + ".xmf").c_str());

#ifdef LIBMESH_HAVE_HDF5
  // flush hdf5 file contents to disk
  if (_enable_hdf5)
    H5Fflush(_hdf5_file_id, H5F_SCOPE_GLOBAL);
#endif

  // increment frame
  _frame++;
}

torch::Tensor
XDMFTensorOutput::extendTensor(torch::Tensor tensor)
{
  // for nodal data we increase each dimension by one and fill in a copy of the slice at 0
  torch::Tensor first;
  using torch::indexing::Slice;

  if (_dim == 3)
  {
    first = tensor.index({0, Slice(), Slice()}).unsqueeze(0);
    tensor = torch::cat({tensor, first}, 0);
    first = tensor.index({Slice(), 0, Slice()}).unsqueeze(1);
    tensor = torch::cat({tensor, first}, 1);
    first = tensor.index({Slice(), Slice(), 0}).unsqueeze(2);
    tensor = torch::cat({tensor, first}, 2);
  }

  else if (_dim == 2)
  {
    first = tensor.index({0}).unsqueeze(0);
    tensor = torch::cat({tensor, first}, 0);
    first = tensor.index({Slice(), 0}).unsqueeze(1);
    tensor = torch::cat({tensor, first}, 1);
  }
  else
    mooseError("Unsupported tensor dimension");

  return tensor.contiguous();
}

#ifdef LIBMESH_HAVE_HDF5
namespace
{
void
addDataToHDF5(hid_t file_id,
              const std::string & dataset_name,
              const char * data,
              std::vector<std::size_t> & ndim,
              hid_t type)
{
  hid_t dataset_id, dataspace_id, plist_id;
  herr_t status;

  // Open the file in read/write mode, create if it doesn't exist

  // hsize_t chunk_dims[RANK];
  std::vector<hsize_t> dims(ndim.begin(), ndim.end());

  // Check if the dataset already exists
  if (H5Lexists(file_id, dataset_name.c_str(), H5P_DEFAULT) > 0)
    mooseError("Dataset '", dataset_name, "' already exists in HDF5 file.");

  // Create a new dataset
  dataspace_id = H5Screate_simple(dims.size(), dims.data(), nullptr);
  if (dataspace_id < 0)
    mooseError("Error creating dataspace");

  plist_id = H5Pcreate(H5P_DATASET_CREATE);
  if (plist_id < 0)
    mooseError("Error creating property list");

  status = H5Pset_chunk(plist_id, dims.size(), dims.data());
  if (status < 0)
    mooseError("Error setting chunking");

  if (H5Zfilter_avail(H5Z_FILTER_DEFLATE))
  {
    unsigned filter_info;
    H5Zget_filter_info(H5Z_FILTER_DEFLATE, &filter_info);
    if (filter_info & H5Z_FILTER_CONFIG_ENCODE_ENABLED)
    {
      status = H5Pset_deflate(plist_id, 9);
      if (status < 0)
        mooseError("Error setting compression filter");
    }
  }

  dataset_id = H5Dcreate(
      file_id, dataset_name.c_str(), type, dataspace_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
  if (dataset_id < 0)
  {
    mooseInfo(dataset_id,
              ' ',
              file_id,
              ' ',
              dataset_name.c_str(),
              ' ',
              type,
              ' ',
              dataspace_id,
              ' ',
              H5P_DEFAULT,
              ' ',
              plist_id,
              ' ',
              H5P_DEFAULT);
    mooseError("Error creating dataset");
  }

  // Write data to the dataset
  status = H5Dwrite(dataset_id, type, H5S_ALL, dataspace_id, H5P_DEFAULT, data);

  // Close resources
  H5Pclose(plist_id);
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
}
}
#endif
