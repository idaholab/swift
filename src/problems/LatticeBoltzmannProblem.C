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

#include "TensorSolver.h"
#include "TensorOperatorBase.h"
#include "TensorTimeIntegrator.h"
#include "TensorOutput.h"
#include "DomainAction.h"

#include "SwiftUtils.h"
#include "DependencyResolverInterface.h"

registerMooseObject("SwiftApp", LatticeBoltzmannProblem);

InputParameters
LatticeBoltzmannProblem::validParams()
{
  InputParameters params = TensorProblem::validParams();
  params.addParam<bool>("is_binary_media", false, "Is binary media provided");
  params.addParam<std::string>(
      "binary_media", "", "The tensor buffer object containing binary media/mesh");
  params.addParam<bool>("enable_slip", false, "Enable slip model");
  // params.addParam<Real>("mfp", 0.0, "Mean free path of the system, (meters)");
  // params.addParam<Real>("dx", 0.0, "Domain resolution, (meters)");
  params.addParam<unsigned int>("substeps", 1, "Number of LBM iterations for every MOOSE timestep");
  params.addParam<Real>("tolerance", 0.0, "LBM convergence tolerance");
  params.addClassDescription("Problem object to enable solving lattice Boltzmann problems");

  return params;
}

LatticeBoltzmannProblem::LatticeBoltzmannProblem(const InputParameters & parameters)
  : TensorProblem(parameters),
    _is_binary_media(getParam<bool>("is_binary_media")),
    _enable_slip(getParam<bool>("enable_slip")),
    /*_mfp(getParam<Real>("mfp")),
    _dx(getParam<Real>("dx")),*/
    _lbm_substeps(getParam<unsigned int>("substeps")),
    _tolerance(getParam<Real>("tolerance"))
{
  // fix sizes
  std::vector<int64_t> shape(_domain.getShape().begin(), _domain.getShape().end());
  if (_domain.getDim() < 3)
    shape.push_back(1);

  for (int64_t i = 0; i < shape.size(); i++)
  {
    _shape_extended.push_back(shape[i]);
    _shape_extended_to_q.push_back(shape[i]);
  }

  // compute unit conversion constants (must happen before compute object::init)
  // const Real & dx = getConstant<Real>("dx");
  // const Real & nu_lu = 1.0 / 3.0 * getConstant<Real>("tau") - 0.5;
  // const Real & Ct = dx * dx * nu_lu / getConstant<Real>("nu");
  // const Real & Cm = getConstant<Real>("rho") * dx * dx * dx;
  // const Real & Cu = dx / Ct;
  // const Real & Crho = getConstant<Real>("rho");
  // declareConstant("C_t", Ct);
  // declareConstant("C_m", Cm);
  // declareConstant("C_Ux", Cu);
  // declareConstant("C_Uy", Cu);
  // declareConstant("C_Uz", Cu);
  // declareConstant("C_U", Cu);
  // declareConstant("C_rho", Crho);
}

void
LatticeBoltzmannProblem::init()
{
  //
  unsigned int n_threads = libMesh::n_threads();
  if (n_threads != 1)
  {
    mooseInfo("Setting libTorch to use ", n_threads, " threads on the CPU.");
    torch::set_num_threads(n_threads);
  }

  // initialize tensors (assuming all scalar for now, but in the future well have an
  // TensorBufferBase pointer as well)
  for (auto pair : _tensor_buffer)
    pair.second->init();

  // compute grid dependent quantities
  gridChanged();

  // init computes (must happen before dependency update)
  for (auto & initializer : _ics)
    initializer->init();

  // init computes (must happen before dependency update)
  for (auto & cmp : _computes)
    cmp->init();

  // update dependencies
  if (_solver)
    _solver->updateDependencies();

  // dependency resolution of TensorICs
  DependencyResolverInterface::sort(_ics);

  // dependency resolution of TensorComputes
  DependencyResolverInterface::sort(_computes);

  // dependency resolution of boundary conditions
  DependencyResolverInterface::sort(_bcs);

  // dependency resolution of Tensor Postprocessors
  DependencyResolverInterface::sort(_pps);

  // show computes
  if (_debug)
  {
    _console << COLOR_CYAN << "Compute object execution order:\n" << COLOR_DEFAULT;
    for (auto & cmp : _computes)
    {
      _console << "  " << cmp->name() << '\n' << COLOR_YELLOW;
      for (const auto & ri : cmp->getRequestedItems())
        _console << "    <- " << ri << '\n';
      _console << COLOR_GREEN;
      for (const auto & si : cmp->getSuppliedItems())
        _console << "    -> " << si << '\n';
      _console << COLOR_DEFAULT;
    }
  }

  // call base class init
  FEProblem::init();

  // init outputs
  for (auto & output : _outputs)
    output->init();

  // commenting this temporarily skips "Unsupported variable type for mapping" error
  // updateDOFMap();

  // debug output
  std::string variable_mapping;
  for (const auto & [buffer_name, tuple] : _buffer_to_var)
    variable_mapping += (std::get<bool>(tuple) ? "NODAL     " : "ELEMENTAL ") + buffer_name + '\n';
  if (!variable_mapping.empty())
    mooseInfo("Direct buffer to solution vector mappings:\n", variable_mapping);

  // fix mesh if provided
  if (_is_binary_media)
    _binary_media = getBuffer(getParam<std::string>("binary_media"));
  else
    _binary_media = torch::ones(_shape, MooseTensor::intTensorOptions());
}

void
LatticeBoltzmannProblem::execute(const ExecFlagType & exec_type)
{
  if (_convergence_residual < _tolerance)
    return;

  if (exec_type == EXEC_INITIAL)
  {
    // update time
    _sub_time = FEProblem::time();
    executeTensorInitialConditions();
    executeTensorOutputs(EXEC_INITIAL);
  }

  if (exec_type == EXEC_TIMESTEP_BEGIN && timeStep() > 1)
  {
    if (dt() != dtOld())
      for (auto & pair : _tensor_buffer)
        pair.second->clearStates();

    // update substepping dt
    _sub_dt = dt() / _lbm_substeps;

    for (unsigned substep = 0; substep < _lbm_substeps; ++substep)
    {
      // create old state buffers
      advanceState();

      // run solver for streaming
      _solver->computeBuffer();

      // run bcs
      for (auto & bc : _bcs)
        bc->computeBuffer();

      // run computes
      for (auto & cmp : _computes)
        cmp->computeBuffer();
      _console << COLOR_WHITE << "Lattice Boltzmann Substep " << substep << ", Residual "
               << _convergence_residual << COLOR_DEFAULT << std::endl;

      _t_total++;
    }

    // run postprocessing before output
    for (auto & pp : _pps)
      pp->computeBuffer();

    // run outputs
    executeTensorOutputs(EXEC_TIMESTEP_BEGIN);

    // mapBuffersToAux();
  }
  FEProblem::execute(exec_type);
}

void
LatticeBoltzmannProblem::addTensorBoundaryCondition(const std::string & compute_type,
                                                    const std::string & name,
                                                    InputParameters & parameters)
{
  addTensorCompute(compute_type, name, parameters, _bcs);
}

void
LatticeBoltzmannProblem::addStencil(const std::string & stencil_name,
                                    const std::string & name,
                                    InputParameters & parameters)
{
  if (_stencil_counter > 0)
    mooseError("Problem object LatticeBoltzmannProblem can only have one stencil");
  // Create the object
  _stencil = _factory.create<LatticeBoltzmannStencilBase>(stencil_name, name, parameters, 0);
  _stencil_counter++;
  logAdd("LatticeBoltzmannStencilBase", name, stencil_name, parameters);

  _shape_extended_to_q.push_back(_stencil->_q);
}

void
LatticeBoltzmannProblem::enableSlipModel()
{

  // const bool & is_mesh_vtk = _lbm_mesh->isMeshVTKFile();

  // if (!is_mesh_vtk)
  //   mooseError("Knudsen and local pore size distributions must be provided with vtk file");

  // else
  // {
  //   // shape of relaxation matrix
  //   std::array<int64_t, 5> extended_shape = {_n[0], _n[1], _n[2], _stencil->_q, _stencil->_q};

  //   // saves relaxation matrix for every mesh element in the domain
  //   _slip_relaxation_matrix = torch::zeros(extended_shape, _options);

  //   // retrieve Knudsen and Local pore sizes
  //   const auto & Kn = _lbm_mesh->getKn();
  //   const auto & pore_size = _lbm_mesh->getPoreSize();

  //   // compute relaxation matrix
  //   for (int i = 0; i < _shape[0]; i++)
  //     for (int j = 0; j < _shape[1]; j++)
  //       for (int k = 0; k < _shape[2]; k++)
  //       {
  //         Real pore_size_scalar = pore_size[i][j][k].item<Real>();
  //         Real kn_scalar = Kn[i][j][k].item<Real>();

  //         Real tau_s = 0.5 + sqrt(6.0 / M_PI) * pore_size_scalar * kn_scalar / (1 + 2 *
  //         kn_scalar); Real tau_d = 0.5 + (3.0 / 2.0) * sqrt(3.0) * 1.0 / pow((1.0 /
  //                     sqrt((_mfp / _dx * 1.0 / (1.0 + 2.0 * kn_scalar))) * 2.0), 2.0);
  //         Real tau_q = 0.5 + (3.0 + M_PI * (2.0 * tau_s - 1.0) * (2.0 * tau_s - 1.0) * _A_1) /
  //         (8.0 * (2.0 * tau_s - 1.0)); torch::Tensor  relaxation_matrix =
  //         torch::diag(torch::tensor({1.0 / 1.0, 1.0 / 1.1, 1.0 / 1.2, 1.0 / tau_d, 1.0 / tau_q,
  //                         1.0 / tau_d, 1.0 / tau_q,  1.0 / tau_s,  1.0 / tau_s}, _options));

  //         _slip_relaxation_matrix.index_put_({i, j, k}, relaxation_matrix);
  //       }
  // }
}

void
LatticeBoltzmannProblem::maskedFillSolids(torch::Tensor & t, const Real & value)
{
  const auto tensor_shape = t.sizes();
  if (_is_binary_media)
  {
    if (t.dim() == _binary_media.dim())
    {
      // 3D
      const auto solid_mask = (_binary_media == value);
      t.masked_fill_(solid_mask, value);
    }
    else
    {
      // 2D and 1D
      const auto solid_mask = (_binary_media == value).unsqueeze(-1).expand(tensor_shape);
      t.masked_fill_(solid_mask, value);
    }
  }
}

void
LatticeBoltzmannProblem::printBuffer(const torch::Tensor & t,
                                     const unsigned int & precision,
                                     const unsigned int & index)
{
  /**
   * Print the entire field for debugging
   */
  torch::Tensor field = t;
  // for buffers higher than 3 dimensions, such as distribution functions
  // select a direction to print or call this method repeatedly to print all directions
  // higher than 4 dimensions is not supported
  // why would one need 5 dimensional tensor for LBM anyway??
  if (t.dim() == 4)
    field = t.select(3, index);

  // This is a sign that the buffer lacks extra dimensions of size 1
  if (t.dim() < 3)
    mooseError("Selected buffer is not 3 dimensional.");

  for (int64_t i = 0; i < field.size(2); i++)
  {
    for (int64_t j = 0; j < field.size(1); j++)
    {
      for (int64_t k = 0; k < field.size(0); k++)
        std::cout << std::fixed << std::setprecision(precision) << field[k][j][i].item<Real>()
                  << " ";
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}
