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
  const Real & dx = getConstant<Real>("dx");
  const Real & nu_lu = 1.0 / 3.0 * getConstant<Real>("tau") - 0.5;
  const Real & Ct = dx * dx * nu_lu / getConstant<Real>("nu");
  const Real & Cm = getConstant<Real>("rho") * dx * dx * dx;
  const Real & Cu = dx / Ct;
  const Real & Crho = getConstant<Real>("rho");
  declareConstant("C_t", Ct);
  declareConstant("C_m", Cm);
  declareConstant("C_Ux", Cu);
  declareConstant("C_Uy", Cu);
  declareConstant("C_Uz", Cu);
  declareConstant("C_U", Cu);
  declareConstant("C_rho", Crho);
}

void
LatticeBoltzmannProblem::init()
{
  TensorProblem::init();

  // fix mesh if provided
  if (_is_binary_media)
    _binary_media = getBuffer(getParam<std::string>("binary_media"));
  else
    _binary_media = torch::ones(_shape, MooseTensor::intTensorOptions());

  // dependency resolution of boundary conditions
  DependencyResolverInterface::sort(_bcs);
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

  _shape_extended_to_q.push_back(_stencil->_q);
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
