/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LBMTensorBuffer.h"
#include "DomainAction.h"
#include "LatticeBoltzmannStencilBase.h"
#include "LatticeBoltzmannProblem.h"

registerMooseObject("SwiftApp", LBMTensorBuffer);

InputParameters
LBMTensorBuffer::validParams()
{
  InputParameters params = TensorBuffer<torch::Tensor>::validParams();
  params.addRequiredParam<std::string>("buffer_type",
                                       "The buffer type can be either distribution function (df), "
                                       "macroscopic scaler (ms) or macroscopic vectorial (mv)");

  params.addParam<bool>("read_from_file", false, "Should the tensor buffer be read from file");
  params.addParam<std::string>("file", "", "Full path of the file to read");

  params.addParam<bool>("is_integer", false, "Whether to specify integer dtype");
  params.addPrivateParam<TensorProblem *>("_tensor_problem", nullptr);
  params.addClassDescription("Tensor wrapper form LBM tensors");

  return params;
}

LBMTensorBuffer::LBMTensorBuffer(const InputParameters & parameters)
  : TensorBuffer<torch::Tensor>(parameters),
    _buffer_type(getParam<std::string>("buffer_type")),
    _lb_problem(dynamic_cast<LatticeBoltzmannProblem &>(
        *getCheckedPointerParam<TensorProblem *>("_tensor_problem"))),
    _stencil(_lb_problem.getStencil())
{
}

void
LBMTensorBuffer::init()
{
  int64_t dimension = 0;
  if (_buffer_type == "df")
    dimension = _stencil._q;
  else if (_buffer_type == "mv")
    dimension = _domain.getDim();
  else if (_buffer_type == "ms")
    dimension = 0;
  else
    mooseError("Buffer type ", _buffer_type, " is not recognized");

  std::vector<int64_t> shape(_domain.getShape().begin(), _domain.getShape().end());

  if (_domain.getDim() < 3)
    shape.push_back(1);
  if (dimension > 0)
    shape.push_back(static_cast<int64_t>(dimension));

  if (getParam<bool>("is_integer"))
    _u = torch::zeros(shape, MooseTensor::intTensorOptions());
  else
    _u = torch::zeros(shape, MooseTensor::floatTensorOptions());

  if (getParam<bool>("read_from_file"))
    readTensorFromFile(shape);
}

void
LBMTensorBuffer::readTensorFromFile(const std::vector<int64_t> & shape)
{
  const std::string tensor_file = getParam<std::string>("file");
  mooseInfo("Loading tensor(s) from file \n" + tensor_file);
  std::ifstream file(tensor_file);
  if (!file.is_open())
    mooseError("Cannot open file " + tensor_file);

  // read mesh into standart vector
  std::vector<int> fileData(shape[0] * shape[1] * shape[2]);
  for (int i = 0; i < fileData.size(); i++)
  {
    if (!(file >> fileData[i]))
    {
      mooseError("Insufficient data in the mesh file");
    }
  }
  file.close();

  // reshape and write into torch tensor
  for (int64_t k = 0; k < shape[2]; k++)
    for (int64_t j = 0; j < shape[1]; j++)
      for (int64_t i = 0; i < shape[0]; i++)
        _u.index_put_({i, j, k}, fileData[k * shape[1] * shape[0] + j * shape[0] + i]);
}

void
LBMTensorBuffer::makeCPUCopy()
{
  if (!_u.defined())
    return;

  if (_cpu_copy_requested)
  {
    if (_u.is_cpu())
      _u_cpu = _u.clone().contiguous();
    else
      _u_cpu = _u.cpu().contiguous();
  }
}
