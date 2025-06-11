/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorProblem.h"

class LatticeBoltzmannStencilBase;
class LatticeBoltzmannMesh;

/**
 * Problem object for solving lattice Boltzmann problems
 */
class LatticeBoltzmannProblem : public TensorProblem
{
public:
  static InputParameters validParams();

  LatticeBoltzmannProblem(const InputParameters & parameters);

  void addTensorBoundaryCondition(const std::string & compute_name,
                                  const std::string & name,
                                  InputParameters & parameters);
  
  // setup stuff
  void init() override;

  // main loop
  void execute(const ExecFlagType & exec_type) override;

  void addStencil(const std::string & stencil_name,
                  const std::string & name,
                  InputParameters & parameters);

  const LatticeBoltzmannStencilBase & getStencil() const {return *_stencil; }

  const bool & isSlipEnabled() const {return _enable_slip; }

  const torch::Tensor & getSlipRelaxationMatrix() const {return _slip_relaxation_matrix;}

  const int & getTotalSteps() const {return _t_total;}
  
  const std::array<int64_t, 3> & getGridSize() const {return _n;}

  /// sets up slip model
  void enableSlipModel();

  /// sets convergence residual
  void setSolverResidual(const Real & residual) {_convergence_residual = residual;};

  /// sets tensor to a value (normally zeros) at solid nodes
  void maskedFillSolids(torch::Tensor & t, const Real & value);

  /// prints the tensor buffer, good for debugging
  void printBuffer(const torch::Tensor & t, const unsigned int & precision, const unsigned int & index);

protected:
  /// LBM Mesh object
  LatticeBoltzmannMesh * _lbm_mesh;

  /// LBM stencils object
  std::shared_ptr<LatticeBoltzmannStencilBase> _stencil;

  /// bc objects
  TensorComputeList _bcs;
  
  /// enables slip models
  bool _enable_slip;

  /// slip coefficient
  const Real _A_1 = 0.6;
  const Real _A_2 = 0.9;

  /// relaxation matrix as a funcion of Kn and local pore size in slip model
  torch::Tensor _slip_relaxation_matrix;

  /// used to restrict construction of lbm stencils to only one
  unsigned int _stencil_counter = 0;

  /// convergence residual
  Real _convergence_residual = 100;

  /// total number of time steps taken
  int _t_total = 0;

  /// lbm substeps
  const unsigned int _lbm_substeps;

  /// lbm convergence tolerance
  const Real _tolerance;

public:
  /// LBM constants
  const Real _cs = 1.0 / sqrt(3.0);
  const Real _cs2 = _cs * _cs;
  const Real _cs4 = _cs2 * _cs2;
};
