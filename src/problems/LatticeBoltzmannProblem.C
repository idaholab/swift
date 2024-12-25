/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LatticeBoltzmannProblem.h"

registerMooseObject("SwiftApp", LatticeBoltzmannProblem);

InputParameters
LatticeBoltzmannProblem::validParams()
{
  InputParameters params = TensorProblem::validParams();
  params.addClassDescription(
      "Problem object to enable solving lattice Boltzmann problems");

  return params;
}

LatticeBoltzmannProblem::LatticeBoltzmannProblem(const InputParameters & parameters)
    : TensorProblem(parameters)
{
}

void
LatticeBoltzmannProblem::addStencil(
                            const std::string & stencil_name,
                             const std::string & name,
                             InputParameters & parameters)
{
}
