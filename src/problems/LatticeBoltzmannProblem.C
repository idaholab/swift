/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannMesh.h"
#include "LatticeBoltzmannStencilBase.h"

#include "TensorOperatorBase.h"
#include "TensorTimeIntegrator.h"
#include "TensorOutput.h"

#include "SwiftUtils.h"
#include "DependencyResolverInterface.h"

registerMooseObject("SwiftApp", LatticeBoltzmannProblem);

InputParameters
LatticeBoltzmannProblem::validParams()
{
  InputParameters params = TensorProblem::validParams();
  params.addParam<bool>("enable_slip", false, "Enable slip model");
  params.addParam<Real>("mfp", 0.0, "Mean free path of the system, (meters)");
  params.addParam<Real>("dx", 0.0, "Domain resolution, (meters)");
  params.addClassDescription(
      "Problem object to enable solving lattice Boltzmann problems");

  return params;
}

LatticeBoltzmannProblem::LatticeBoltzmannProblem(const InputParameters & parameters)
    : TensorProblem(parameters),
    _lbm_mesh(dynamic_cast<LatticeBoltzmannMesh *>(&_mesh)),
    _enable_slip(getParam<bool>("enable_slip")),
    _mfp(getParam<Real>("mfp")),
    _dx(getParam<Real>("dx"))
{
}

void
LatticeBoltzmannProblem::init()
{
  // initialize base class method
  TensorProblem::init();

  // initialize buffers with extra dimensions
  for (auto & pair : _tensor_buffer)
  {
    auto extra_dim = _buffer_extra_dimension.find(pair.first);

    if (extra_dim->second >= 1)
    {
      // buffers with extra dimension
      unsigned int dim = 4;
      std::array<int64_t, 4> n;
      std::fill(n.begin(), n.end(), 1);
      std::copy(_n.begin(), _n.end(), n.begin());
      n[dim - 1] = static_cast<int64_t>(extra_dim->second);
      torch::IntArrayRef shape = torch::IntArrayRef(n.data(), dim);
      pair.second = torch::zeros(shape, _options);
    }
  }

  // set up parameters for slip flow
  if (_enable_slip)
    enableSlipModel();
}

void
LatticeBoltzmannProblem::execute(const ExecFlagType & exec_type)
{ 
  /**
   * This is primarily a copy of base class execute function with a 
   * different order order of operations in the main loop.
   */
  if (exec_type == EXEC_INITIAL)
  {
    // run ICs
    for (auto & ic : _ics)
      ic->computeBuffer();

    // compile ist of compute output tensors
    std::set<std::string> _is_output;
    for (auto & cmp : _computes)
      _is_output.insert(cmp->getSuppliedItems().begin(), cmp->getSuppliedItems().end());

    // check for uninitialized tensors
    for (auto & [name, t] : _tensor_buffer)
      if (!t.defined() && _is_output.count(name) == 0)
        mooseWarning(name, " is not initialized and not an output of any [Solve] compute.");
  }

  if (exec_type == EXEC_TIMESTEP_BEGIN)
  {
    _convergence_residual = 1.0;
    if (dt() != dtOld())
      for (auto & [name, max_states] : _old_tensor_buffer)
        max_states.second.clear();

    // update substepping dt
    _sub_dt = dt() / _substeps;

    for (unsigned substep = 0; substep < _substeps; ++substep)
    {
      // create old state buffers
      advanceState();

      // run timeintegrators
      for (auto & ti : _time_integrators)
        ti->computeBuffer();

      // run bcs
      for (auto & bc : _bcs)
        bc->computeBuffer();

      // run computes
      for (auto & cmp : _computes)
        cmp->computeBuffer();
      _console << COLOR_WHITE << "Lattice Boltzmann Substep "<< substep<<", Residual "<<_convergence_residual << COLOR_DEFAULT << std::endl;

      _t_total ++;
    }

    // run postprocessing before output
    for (auto & pp : _pps)
      pp->computeBuffer();

    // wait for prior asynchronous activity on CPU buffers to complete
    // (this is a synchronization barrier for the threaded CPU activity)
    for (auto & output : _outputs)
      output->waitForCompletion();

    // prepare CPU buffers (this is a synchronization barrier for the GPU)
    for (auto & [name, cpu_buffer] : _tensor_cpu_buffer)
    {
      // get main buffer (GPU or CPU) - we already verified that it must exist
      const auto & buffer = _tensor_buffer[name];
      if (buffer.is_cpu())
        cpu_buffer = buffer.clone().contiguous();
      else
        cpu_buffer = buffer.cpu().contiguous();
    }

    // run direct buffer outputs (asynchronous in threads)
    for (auto & output : _outputs)
      output->startOutput();

    mapBuffersToAux();
  }
  FEProblem::execute(exec_type);
}

void
LatticeBoltzmannProblem::addTensorBuffer(const std::string & buffer_name, InputParameters & parameters)
{
  // run base class method
  TensorProblem::addTensorBuffer(buffer_name, parameters);

  // initialize
  _buffer_extra_dimension[buffer_name] = 0;

  // add extra dimension if necessary
  if(parameters.isParamValid("vector_size"))
    _buffer_extra_dimension[buffer_name] = parameters.get<unsigned int>("vector_size");
}

void
LatticeBoltzmannProblem::addStencil(
                            const std::string & stencil_name,
                             const std::string & name,
                             InputParameters & parameters)
{
  if (_stencil_counter > 0)
    mooseError("Problem object LatticeBoltzmannProblem can only have one stencil");
  // Create the object
  _stencil = _factory.create<LatticeBoltzmannStencilBase>(stencil_name, name, parameters, 0);
  _stencil_counter ++;
  logAdd("LatticeBoltzmannStencilBase", name, stencil_name, parameters);
}

void
LatticeBoltzmannProblem::enableSlipModel()
{

  const bool & is_mesh_vtk = _lbm_mesh->isMeshVTKFile();

  if (!is_mesh_vtk)
    mooseError("Knudsen and local pore size distributions must be provided with vtk file");

  else
  {
    // shape of relaxation matrix
    std::array<int64_t, 5> extended_shape = {_n[0], _n[1], _n[2], _stencil->_q, _stencil->_q};

    // saves relaxation matrix for every mesh element in the domain
    _slip_relaxation_matrix = torch::zeros(extended_shape, _options);

    // retrieve Knudsen and Local pore sizes
    const auto & Kn = _lbm_mesh->getKn();
    const auto & pore_size = _lbm_mesh->getPoreSize();

    // compute relaxation matrix
    for (int i = 0; i < _shape[0]; i++)
      for (int j = 0; j < _shape[1]; j++)
        for (int k = 0; k < _shape[2]; k++)
        {
          Real pore_size_scalar = pore_size[i][j][k].item<Real>();
          Real kn_scalar = Kn[i][j][k].item<Real>();

          Real tau_s = 0.5 + sqrt(6.0 / M_PI) * pore_size_scalar * kn_scalar / (1 + 2 * kn_scalar);
          Real tau_d = 0.5 + (3.0 / 2.0) * sqrt(3.0) * 1.0 / pow((1.0 /
                      sqrt((_mfp / _dx * 1.0 / (1.0 + 2.0 * kn_scalar))) * 2.0), 2.0);
          Real tau_q = 0.5 + (3.0 + M_PI * (2.0 * tau_s - 1.0) * (2.0 * tau_s - 1.0) * _A) / (8.0 * (2.0 * tau_s - 1.0));
          torch::Tensor  relaxation_matrix = torch::diag(torch::tensor({1.0 / 1.0, 1.0 / 1.1, 1.0 / 1.2, 1.0 / tau_d, 1.0 / tau_q,
                          1.0 / tau_d, 1.0 / tau_q,  1.0 / tau_s,  1.0 / tau_s}, _options));

          _slip_relaxation_matrix.index_put_({i, j, k}, relaxation_matrix);
        }
  }
}

void
LatticeBoltzmannProblem::setSolverResidual(const Real & residual)
{
  _convergence_residual = residual;
}

void
LatticeBoltzmannProblem::setTensorToValue(torch::Tensor & t, const Real & value)
{
  auto tensor_shape = t.sizes();
  const bool & is_mesh_file = _lbm_mesh->isMeshDatFile();
  const bool & is_mesh_vtk = _lbm_mesh->isMeshVTKFile();
  const auto & binary_mesh = _lbm_mesh->getBinaryMesh();
  if (is_mesh_file || is_mesh_vtk)
  {
    if (t.dim() == binary_mesh.dim())
    {
      // 3D
      const auto solid_mask = (binary_mesh == value);
      t.masked_fill_(solid_mask, value);
    }
    else
    {
      // 2D and 1D
      const auto solid_mask = (binary_mesh == value).unsqueeze(-1).expand(tensor_shape);
      t.masked_fill_(solid_mask, value);
    }
  }
}
