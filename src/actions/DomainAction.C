/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "DomainAction.h"
#include "TensorProblem.h"
#include "MooseEnum.h"
#include "SetupMeshAction.h"
#include "SwiftApp.h"

#include <initializer_list>

// run this early, before any objects are constructed
registerMooseAction("SwiftApp", DomainAction, "meta_action");
registerMooseAction("SwiftApp", DomainAction, "add_mesh_generator");

InputParameters
DomainAction::validParams()
{
  InputParameters params = Action::validParams();
  params.addClassDescription("Set up the domain and compute devices.");

  MooseEnum dims("1=1 2 3");
  params.addRequiredParam<MooseEnum>("dim", dims, "Problem dimension");

  MooseEnum parmode("NONE FFT_SLAB FFT_PENCIL", "NONE");
  parmode.addDocumentation("NONE", "Serial execution without domain decomposition.");
  parmode.addDocumentation("FFT_SLAB",
                           "Slab decomposition with X-Z slabs stacked along the Y direction in "
                           "real space and Y-Z slabs stacked along the X direction in Fourier "
                           "space. This requires one all-to-all communication per FFT.");
  parmode.addDocumentation(
      "FFT_PENCIL",
      "Pencil decomposition (3D only). Three 1D FFTs in pencil arrays along the X, Y, and lastly Z "
      "direction. Thie requires two many-to-many communications per FFT.");

  params.addParam<MooseEnum>("parallel_mode", parmode, "Parallelization mode.");

  params.addParam<unsigned int>("nx", 1, "Number of elements in the X direction");
  params.addParam<unsigned int>("ny", 1, "Number of elements in the Y direction");
  params.addParam<unsigned int>("nz", 1, "Number of elements in the Z direction");
  params.addParam<Real>("xmax", 1.0, "Upper X Coordinate of the generated mesh");
  params.addParam<Real>("ymax", 1.0, "Upper Y Coordinate of the generated mesh");
  params.addParam<Real>("zmax", 1.0, "Upper Z Coordinate of the generated mesh");
  params.addParam<Real>("xmin", 0.0, "Lower X Coordinate of the generated mesh");
  params.addParam<Real>("ymin", 0.0, "Lower Y Coordinate of the generated mesh");
  params.addParam<Real>("zmin", 0.0, "Lower Z Coordinate of the generated mesh");

  MooseEnum meshmode("DUMMY DOMAIN MANUAL", "DUMMY");
  meshmode.addDocumentation("DUMMY",
                            "Create a single element mesh the size of the simulation domain");
  meshmode.addDocumentation("DOMAIN", "Create a mesh with one element per grid cell");
  meshmode.addDocumentation("MANUAL",
                            "Do not auto-generate a mesh. User must add a Mesh block themselves.");

  params.addParam<MooseEnum>("mesh_mode", meshmode, "Mesh generation mode.");

  params.addRequiredParam<std::vector<std::string>>("device_names", "Compute devices to run on.");
  params.addParam<std::vector<unsigned int>>(
      "device_weights", {}, "Device weights (or speeds) to influence the partitioning.");
  return params;
}

DomainAction::DomainAction(const InputParameters & parameters)
  : Action(parameters),
    _device_names(getParam<std::vector<std::string>>("device_names")),
    _device_weights(getParam<std::vector<unsigned int>>("device_weights")),
    _parallel_mode(getParam<MooseEnum>("parallel_mode").getEnum<ParallelMode>()),
    _dim(getParam<MooseEnum>("dim")),
    _n_global(
        {getParam<unsigned int>("nx"), getParam<unsigned int>("ny"), getParam<unsigned int>("nz")}),
    _min_global({getParam<Real>("xmin"), getParam<Real>("ymin"), getParam<Real>("zmin")}),
    _max_global({getParam<Real>("xmax"), getParam<Real>("ymax"), getParam<Real>("zmax")}),
    _mesh_mode(getParam<MooseEnum>("mesh_mode").getEnum<MeshMode>()),
    _shape(torch::IntArrayRef(_n_local.data(), _dim)),
    _reciprocal_shape(torch::IntArrayRef(_n_reciprocal_local.data(), _dim)),
    _rank(_communicator.rank()),
    _n_rank(_communicator.size()),
    _send_tensor(_n_rank),
    _recv_tensor(_n_rank)
{
  if (_parallel_mode == ParallelMode::NONE && comm().size() > 1)
    paramError("parallel_mode", "NONE requires the application to run in serial.");

  // process weights
  if (_device_weights.empty())
    _device_weights.assign(1, _device_names.size());

  if (_device_weights.size() != _device_names.size())
    mooseError("Specify one weight per device or none at all");

  // determine the processor name
  char name[MPI_MAX_PROCESSOR_NAME + 1];
  int len;
  MPI_Get_processor_name(name, &len);
  name[len] = 0;

  // gather all processor names
  std::vector<std::string> host_names;
  _communicator.allgather(std::string(name), host_names);

  // get the local rank on the current processor (used for compute device assignment)
  std::map<std::string, unsigned int> host_rank_count;

  for (const auto & host_name : host_names)
  {
    if (host_rank_count.find(name) == host_rank_count.end())
      host_rank_count[host_name] = 0;

    auto & local_rank = host_rank_count[host_name];
    _local_ranks.push_back(local_rank);
    _local_weights.push_back(_device_weights[local_rank % _device_weights.size()]);

    // std::cout << "Process on " << host_name << ' ' << local_rank << ' '
    //           << _device_weights[local_rank % _device_weights.size()] << '\n';

    local_rank++;
  }

  // for (const auto i : index_range(host_names))
  //   std::cout << host_names[i] << '\t' << _local_ranks[i] << '\n';

  // pick a compute device for a list of available devices
  auto swift_app = dynamic_cast<SwiftApp *>(&_app);
  if (!swift_app)
    mooseError("This action requires a SwftApp object to be present.");
  swift_app->setTorchDevice(_device_names[_local_ranks[_rank] % _device_names.size()], {});

  // domain partitioning
  gridChanged();
}

void
DomainAction::gridChanged()
{
  auto options = MooseTensor::floatTensorOptions();

  // build real space axes
  _volume_global = 1.0;
  for (const unsigned int dim : {0, 1, 2})
  {
    // error check
    if (_max_global[dim] <= _min_global[dim])
      mooseError("Max coordinate must be larger than the min coordinate in every dimension");

    // get grid geometry
    _grid_spacing[dim] = (_max_global[dim] - _min_global[dim]) / _n_global[dim];

    // real space axis
    if (dim < _dim)
    {
      _global_axis[dim] =
          align(torch::linspace(c10::Scalar(_min_global[dim] + _grid_spacing[dim] / 2.0),
                                c10::Scalar(_max_global[dim] - _grid_spacing[dim] / 2.0),
                                _n_global[dim],
                                options),
                dim);
      _volume_global *= _max_global[dim] - _min_global[dim];
    }
    else
      _global_axis[dim] = torch::tensor({0.0}, options);
  }

  // build reciprocal space axes
  for (const unsigned int dim : {0, 1, 2})
  {
    if (dim < _dim)
    {
      const auto freq = (dim == _dim - 1)
                            ? torch::fft::rfftfreq(_n_global[dim], _grid_spacing[dim], options)
                            : torch::fft::fftfreq(_n_global[dim], _grid_spacing[dim], options);
      _global_reciprocal_axis[dim] = align(freq * 2.0 * libMesh::pi, dim);
    }
    else
      _global_reciprocal_axis[dim] = torch::tensor({0.0}, options);

    // compute max frequency along each axis
    _max_k[dim] = libMesh::pi / _grid_spacing[dim];

    // get global reciprocal axis size
    _n_reciprocal_global[dim] = _global_reciprocal_axis[dim].sizes()[dim];
  }

  switch (_parallel_mode)
  {
    case ParallelMode::NONE:
      partitionSerial();
      break;

    case ParallelMode::FFT_SLAB:
      partitionSlabs();
      break;

    case ParallelMode::FFT_PENCIL:
      partitionPencils();
      break;
  }

  // get local reciprocal axis size
  for (const auto dim : {0, 1, 2})
    _n_reciprocal_local[dim] = _local_reciprocal_axis[dim].sizes()[dim];

  // k-square buffer
  _k2 = _local_reciprocal_axis[0] * _local_reciprocal_axis[0] +
        _local_reciprocal_axis[1] * _local_reciprocal_axis[1] +
        _local_reciprocal_axis[2] * _local_reciprocal_axis[2];
}

void
DomainAction::partitionSerial()
{
  // goes along the full dimension for each rank
  for (const auto d : make_range(3u))
  {
    _local_begin[d].resize(_n_rank);
    _local_end[d].resize(_n_rank);
    for (const auto i : make_range(_communicator.size()))
    {
      _local_begin[d][i] = 0;
      _local_end[d][i] = _n_global[d];
    }
  }

  // to do, make those slices dependent on local begin/end
  _local_axis = _global_axis;
  _n_local = _n_global;
  _local_reciprocal_axis = _global_reciprocal_axis;
}

void
DomainAction::partitionSlabs()
{
  if (_dim < 2)
    paramError("dim", "Dimension must be 2 or 3 for slab decomposition.");

  // x is partitioned along a halved dimension due to the use of rfft
  _n_local_all[0] = partitionHepler(_global_reciprocal_axis[0].sizes()[0], _device_weights);

  // y is partitioned along the y realspace axis
  _n_local_all[1] = partitionHepler(_global_axis[1].sizes()[1], _device_weights);

  // set begin/end for x and y
  for (const auto d : {0, 1})
  {
    int64_t b = 0;
    for (const auto r: index_range(_n_local_all[d]))
    {
      _local_begin[d][r] = b;
      b += _n_local_all[d][r];
      _local_end[d][r] = b;
    }
  }

  // z is not partitioned at all
  _n_local_all[2].assign(_n_rank, _n_global[2]);
  _local_begin[2].assign(_n_rank, 0);
  _local_end[2].assign(_n_rank, _n_global[2]);

  // slice the real space into x-z slabs stacked in y direction
  _local_axis[0] = _global_axis[0].slice(0, 0, _n_global[0]);
  _local_axis[1] = _global_axis[1].slice(1, _local_begin[1][_rank], _local_end[1][_rank]);
  _n_local[0] = _n_global[0];
  _n_local[1] = _local_end[1][_rank] - _local_begin[1][_rank];

  // slice the reciprocal space into y-z slices stacked in x direction
  _local_reciprocal_axis[0] =
      _global_reciprocal_axis[0].slice(0, 0, _local_begin[0][_rank], _local_end[0][_rank]);
  _local_reciprocal_axis[1] = _global_reciprocal_axis[1].slice(1, 0, _n_reciprocal_global[1]);

  _n_local[2] = _n_global[2];

  // special casing this should not be neccessary
  if (_dim == 3)
  {
    _local_axis[2] = _global_axis[2].slice(2, 0, _n_global[2]);
    _local_reciprocal_axis[2] = _global_reciprocal_axis[2].slice(2, 0, _n_reciprocal_global[2]);
  }
  else
  {
    _local_axis[2] = _global_axis[2];
    _local_reciprocal_axis[2] = _global_reciprocal_axis[2];
  }

  // allocate receive buffer
  for (const auto i : make_range(_communicator.size()))
    if (i != _rank)
      _recv_data[i].resize(_n_local_all[0][_rank] * _n_local_all[1][i] * _n_local_all[2][i]);
}

void
DomainAction::partitionPencils()
{
  if (_dim < 3)
    paramError("dim", "Dimension must be 3 for pencil decomposition.");
  paramError("parallel_mode", "Not implemented yet!");
}

void
DomainAction::act()
{
  if (_current_task == "meta_action" && _mesh_mode != MeshMode::SWIFT_MANUAL)
  {
    // check if a SetupMesh action exists
    auto mesh_actions = _awh.getActions<SetupMeshAction>();
    // for (const auto & ma : mesh_actions)
    //   mooseInfoRepeated(ma->name());
    if (mesh_actions.size() > 0)
      paramError("mesh_mode", "Do not specify a [Mesh] block unless mesh_mode is set to MANUAL");

    // otherwise create one
    auto & af = _app.getActionFactory();
    InputParameters action_params = af.getValidParams("SetupMeshAction");
    auto action = std::static_pointer_cast<MooseObjectAction>(
        af.create("SetupMeshAction", "Mesh", action_params));
    _app.actionWarehouse().addActionBlock(action);
  }

  // add a DomainMeshGenerator
  if (_current_task == "add_mesh_generator" && _mesh_mode != MeshMode::SWIFT_MANUAL)
  {
    // Don't do mesh generators when recovering or when the user has requested for us not to
    if ((_app.isRecovering() && _app.isUltimateMaster()) || _app.masterMesh())
      return;

    const MeshGeneratorName name = "domain_mesh_generator";
    auto params = _factory.getValidParams("DomainMeshGenerator");

    params.set<MooseEnum>("dim") = _dim;
    params.set<Real>("xmax") = _max_global[0];
    params.set<Real>("ymax") = _max_global[1];
    params.set<Real>("zmax") = _max_global[2];
    params.set<Real>("xmin") = _min_global[0];
    params.set<Real>("ymin") = _min_global[1];
    params.set<Real>("zmin") = _min_global[2];

    if (_mesh_mode == MeshMode::SWIFT_DOMAIN)
    {
      params.set<unsigned int>("nx") = _n_global[0];
      params.set<unsigned int>("ny") = _n_global[1];
      params.set<unsigned int>("nz") = _n_global[2];
    }
    else if (_mesh_mode == MeshMode::SWIFT_DUMMY)
    {
      params.set<unsigned int>("nx") = 1;
      params.set<unsigned int>("ny") = 1;
      params.set<unsigned int>("nz") = 1;
    }
    else
      mooseError("Internal error");

    _app.addMeshGenerator("DomainMeshGenerator", name, params);
  }
}

const torch::Tensor &
DomainAction::getAxis(std::size_t component) const
{
  if (component < 3)
    return _local_axis[component];
  mooseError("Invalid component");
}

const torch::Tensor &
DomainAction::getReciprocalAxis(std::size_t component) const
{
  if (component < 3)
    return _local_reciprocal_axis[component];
  mooseError("Invalid component");
}

torch::Tensor
DomainAction::fft(const torch::Tensor & t) const
{
  switch (_parallel_mode)
  {
    case ParallelMode::NONE:
      return fftSerial(t);

    case ParallelMode::FFT_SLAB:
      return fftSlab(t);

    case ParallelMode::FFT_PENCIL:
      return fftPencil(t);
  }
  mooseError("Not implemented");
}

torch::Tensor
DomainAction::fftSerial(const torch::Tensor & t) const
{
  switch (_dim)
  {
    case 1:
      return torch::fft::rfft(t);
    case 2:
      return torch::fft::rfft2(t);
    case 3:
      return torch::fft::rfftn(t, c10::nullopt, {0, 1, 2});
    default:
      mooseError("Unsupported mesh dimension");
  }
}

torch::Tensor
DomainAction::fftSlab(const torch::Tensor & t) const
{
  mooseInfoRepeated("fftSlab");
  if (_dim == 1)
    mooseError("Unsupported mesh dimension");

  MooseTensor::printTensorInfo(t);

  // 2D transform the local slab
  auto slab =
      _dim == 3 ? torch::fft::fft2(t, c10::nullopt, {0, 2}) : torch::fft::fft(t, c10::nullopt, 0);
  MooseTensor::printTensorInfo(slab);

  // send
  std::vector<MPI_Request> send_requests(_n_rank, MPI_REQUEST_NULL);
  for (const auto & i : make_range(_n_rank))
    if (i != _rank)
    {
      _send_tensor[i] = slab.slice(0, _local_begin[0][i], _local_end[0][i]).contiguous().cpu();
      MooseTensor::printTensorInfo(_send_tensor[i]);

      auto data_ptr = _send_tensor[i].data_ptr<double>();
      MPI_Isend(
          data_ptr, _send_tensor[i].numel(), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &send_requests[i]);
    }
    else
      // keep the local slice on device
      _recv_tensor[i] = slab.slice(0, _local_begin[0][i], _local_end[0][i]);

  // receive
  MPI_Status recv_status;
  for (const auto & i : make_range(_n_rank))
    if (i != _rank)
      MPI_Recv(_recv_data[i].data(), 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &recv_status);

  // Wait for all non-blocking sends to complete
  for (const auto & i : make_range(_n_rank))
    if (i != _rank)
    {
      // 2d _n_local_all[0][_rank] * _n_local_all[1][i] * _n_local_all[2][i]
      _recv_tensor[i] =
          torch::from_blob(_recv_data[i].data(),
                           {_n_local_all[0][_rank], _n_local_all[1][i]},
                           torch::kFloat64)
              .to(MooseTensor::floatTensorOptions()); // todo: take care of 32 but floats as well!
    }

  // stack
  auto t2 = torch::vstack(_recv_tensor);

  // Wait for all non-blocking sends to complete
  MPI_Waitall(_n_rank, send_requests.data(), MPI_STATUSES_IGNORE);

  // transfor along y direction
  return torch::fft::rfft(t2, c10::nullopt, 1);
}

torch::Tensor
DomainAction::fftPencil(const torch::Tensor & /*t*/) const
{
  if (_dim != 3)
    mooseError("Unsupported mesh dimension");
  paramError("parallel_mode", "Not implemented yet!");
}

torch::Tensor
DomainAction::ifft(const torch::Tensor & t) const
{
  switch (_dim)
  {
    case 1:
      return torch::fft::irfft(t, getShape()[0]);
    case 2:
      return torch::fft::irfft2(t, getShape());
    case 3:
      return torch::fft::irfftn(t, getShape(), {0, 1, 2});
    default:
      mooseError("Unsupported mesh dimension");
  }
}

torch::Tensor
DomainAction::align(torch::Tensor t, unsigned int dim) const
{
  if (dim >= _dim)
    mooseError("Unsupported alignment dimension requested dimension");

  switch (_dim)
  {
    case 1:
      return t;

    case 2:
      if (dim == 0)
        return torch::unsqueeze(t, 1);
      else
        return torch::unsqueeze(t, 0);

    case 3:
      if (dim == 0)
        return t.unsqueeze(1).unsqueeze(2);
      else if (dim == 1)
        return t.unsqueeze(0).unsqueeze(2);
      else
        return t.unsqueeze(0).unsqueeze(0);

    default:
      mooseError("Unsupported mesh dimension");
  }
}
