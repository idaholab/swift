/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMStream.h"
#include "LatticeBoltzmannProblem.h"
#include "LatticeBoltzmannStencilBase.h"

using namespace torch::indexing;

registerMooseObject("SwiftApp", LBMStream);

InputParameters
LBMStream::validParams()
{
  InputParameters params = TensorSolver::validParams();
  params.addClassDescription("LBM Streaming operation.");
  params.addRequiredParam<TensorOutputBufferName>("buffer",
                                                  "The buffer this compute is writing to");
  params.addRequiredParam<TensorInputBufferName>("f_old", "Old time step distribution");
  return params;
}

LBMStream::LBMStream(const InputParameters & parameters)
  : TensorSolver(parameters),
    _lb_problem(dynamic_cast<LatticeBoltzmannProblem &>(_tensor_problem)),
    _stencil(_lb_problem.getStencil()),
    _u(getOutputBuffer<torch::Tensor>("buffer")),
    _f_old(getBufferOld("f_old", 1))
{
}

void
LBMStream::computeBuffer()
{
  const auto n_old = _f_old.size();
  if (n_old != 0)
  {
    // do not overwrite previous
    _u = _u.clone();
    for (int i = 0; i < _stencil._q; i++)
    {
      _u.index_put_({Slice(), Slice(), Slice(), i},
                    torch::roll(_f_old[0].index({Slice(), Slice(), Slice(), i}),
                                /* shifts = */
                                {_stencil._ex[i].item<int64_t>(),
                                 _stencil._ey[i].item<int64_t>(),
                                 _stencil._ez[i].item<int64_t>()},
                                /* dims = */
                                {0, 1, 2}));
    }
    _lb_problem.maskedFillSolids(_u, 0);
  }
}
