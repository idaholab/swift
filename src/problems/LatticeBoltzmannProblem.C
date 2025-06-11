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
    _lbm_mesh(dynamic_cast<LatticeBoltzmannMesh *>(&_mesh)),
    _enable_slip(getParam<bool>("enable_slip")),
    /*_mfp(getParam<Real>("mfp")),
    _dx(getParam<Real>("dx")),*/
    _lbm_substeps(getParam<unsigned int>("substeps")),
    _tolerance(getParam<Real>("tolerance"))
{
  // set up unit conversion
  Real Cl = _scalar_constants.at("dx");
  Real nu_lu = 1.0 / 3.0 * (_scalar_constants.at("tau") - 0.5);
  Real Ct = nu_lu / _scalar_constants.at("nu") * Cl * Cl;
  Real Cm = _scalar_constants.at("rho") * Cl * Cl * Cl;
  Real Cu = Ct / Cl;

  _scalar_constants.insert(std::pair<std::string, Real>("Ct", Ct));
  _scalar_constants.insert(std::pair<std::string, Real>("Cm", Cm));
  _scalar_constants.insert(std::pair<std::string, Real>("Cu", Cu));
}

void
LatticeBoltzmannProblem::execute(const ExecFlagType & exec_type)
{
  // convergence check
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
}

void
LatticeBoltzmannProblem::maskedFillSolids(torch::Tensor & t, const Real & value)
{
  const auto tensor_shape = t.sizes();
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
