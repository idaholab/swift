/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorHistogram.h"
#include "SwiftUtils.h"

#include <ATen/ATen.h>
#include <ATen/native/Histogram.h>

registerMooseObject("SwiftApp", TensorHistogram);

InputParameters
TensorHistogram::validParams()
{
  InputParameters params = TensorVectorPostprocessor::validParams();
  params.addClassDescription("Compute a histogram of the given tensor.");
  params.addRequiredParam<Real>("min", "Lower bound of the histogram.");
  params.addRequiredParam<Real>("max", "Upper bound of the histogram.");
  params.addRequiredRangeCheckedParam<std::size_t>("bins", "bins>0", "Number of histogram bins.");
  return params;
}

TensorHistogram::TensorHistogram(const InputParameters & parameters)
  : TensorVectorPostprocessor(parameters),
    _min(getParam<Real>("min")),
    _max(getParam<Real>("max")),
    _bins(getParam<std::size_t>("bins")),
    _bin_edges(torch::linspace(_min, _max, _bins + 1, MooseTensor::floatTensorOptions())),
    _bin_vec(declareVector("bin")),
    _count_vec(declareVector("count"))
{
  // error check (if this is not fulfilled teh histogram will be empty)
  if (_min > _max)
    paramError("min", "max must be greater than min");

  // fill the bin vector
  _bin_vec.resize(_bins);
  _count_vec.resize(_bins);
  const auto step = (_max - _min) / _bins;
  for (const auto i : make_range(_bins))
    _bin_vec[i] = _min + step / 2.0 + step * i;
}

void
TensorHistogram::execute()
{
  // Reshape the data to fit the expected input format for histogramdd
  const auto data = _u.reshape({-1, 1});

  // Use the histogramdd function
  const auto pair = at::native::histogramdd(data, {_bin_edges});
  const auto hist = std::get<0>(pair).cpu();

  // put into VPP vector
  if (hist.dtype() == torch::kFloat32)
    for (const auto i : make_range(int64_t(_bins)))
      _count_vec[i] = hist.index({i}).item<float>();
  else if (hist.dtype() == torch::kFloat64)
    for (const auto i : make_range(int64_t(_bins)))
      _count_vec[i] = hist.index({i}).item<double>();
  else
    mooseError("Unsupported tensor dtype() in TensorHistogram.");
}
