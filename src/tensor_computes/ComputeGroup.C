/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ComputeGroup.h"
#include "TensorProblem.h"
#include "SwiftUtils.h"

registerMooseObject("SwiftApp", ComputeGroup);

InputParameters
ComputeGroup::validParams()
{
  InputParameters params = TensorOperatorBase::validParams();
  params.addClassDescription("Group of operators with internal dependency resolution.");
  params.addParam<std::vector<TensorComputeName>>(
      "computes", {}, "List of grouped tensor computes.");
  return params;
}

ComputeGroup::ComputeGroup(const InputParameters & parameters)
  : TensorOperatorBase(parameters), _visited(false)
{
}

void
ComputeGroup::init()
{
  // grab requested computes
  const auto & computes = getParam<std::vector<TensorComputeName>>("computes");
  std::set<TensorComputeName> requested_computes(computes.begin(), computes.end());
  for (const auto & cmp : _tensor_problem.getComputes())
    if (requested_computes.count(cmp->name()))
      _computes.push_back(cmp);
}

void
ComputeGroup::computeBuffer()
{
  for (const auto & cmp : _computes)
    try
    {
      cmp->computeBuffer();
    }
    catch (const std::exception & e)
    {
      cmp->mooseError("Exception: ", e.what());
    }
}

void
ComputeGroup::updateDependencies()
{
  // detect recursive self use
  if (!_visited)
    _visited = true;
  else
    paramError("computes", "Compute is using itself, creating an unresolvable dependency.");

  // recursively update dependencies of the constituent operators
  for (const auto & cmp : _computes)
    cmp->updateDependencies();

  // dependency resolution of TensorComputes
  DependencyResolverInterface::sort(_computes);

  // determine total in/out
  std::set<std::string> in, out;
  for (const auto & cmp : _computes)
  {
    const auto & cin = cmp->getRequestedItems();
    const auto & cout = cmp->getSuppliedItems();
    in.insert(cin.begin(), cin.end());
    out.insert(cout.begin(), cout.end());
  }
  std::set_difference(in.begin(),
                      in.end(),
                      out.begin(),
                      out.end(),
                      std::inserter(_requested_buffers, _requested_buffers.begin()));
  std::set_difference(out.begin(),
                      out.end(),
                      in.begin(),
                      in.end(),
                      std::inserter(_supplied_buffers, _supplied_buffers.begin()));
}
