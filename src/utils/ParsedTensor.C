/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "ParsedTensor.h"
#include "libmesh/extrasrc/fptypes.hh"

ParsedTensor::ParsedTensor() : FunctionParserAD(), _data(*getParserData())
{
  _mFFT = _data.mFuncPtrs.size();
  this->AddFunction("FFT", fp_dummy, 1);
  _miFFT = _data.mFuncPtrs.size();
  this->AddFunction("iFFT", fp_dummy, 1);
}

Real
ParsedTensor::fp_dummy(const Real *)
{
  throw std::runtime_error("This function is only implemented for torch tensors");
}

void
ParsedTensor::setupTensors()
{
  // allocate stack
  s.resize(_data.mStackSize);

  // convert immediate data
  tensor_immed.clear();
  for (const auto & i : _data.mImmed)
    tensor_immed.push_back(torch::tensor(i, MooseTensor::floatTensorOptions()));
}

torch::Tensor
ParsedTensor::Eval(const std::vector<const torch::Tensor *> & params)
{
  using namespace FUNCTIONPARSERTYPES;

  // get a reference to the stored bytecode
  const auto & ByteCode = _data.mByteCode;

  int nImmed = 0, sp = -1, op;
  for (unsigned int i = 0; i < ByteCode.size(); ++i)
  {
    // execute bytecode
    switch (op = ByteCode[i])
    {
      case cImmed:
        ++sp;
        s[sp] = tensor_immed[nImmed++];
        break;
      case cAdd:
        --sp;
        s[sp] = s[sp] + s[sp + 1];
        break;
      case cSub:
        --sp;
        s[sp] = s[sp] - s[sp + 1];
        break;
      case cRSub:
        --sp;
        s[sp] = s[sp + 1] - s[sp];
        break;
      case cMul:
        --sp;
        s[sp] = s[sp] * s[sp + 1];
        break;
      case cDiv:
        --sp;
        s[sp] = s[sp] / s[sp + 1];
        break;
      case cMod:
        --sp;
        s[sp] = fmod(s[sp], s[sp + 1]);
        break;
      case cRDiv:
        --sp;
        s[sp] = s[sp + 1] / s[sp];
        break;

      case cSin:
        s[sp] = sin(s[sp]);
        break;
      case cCos:
        s[sp] = cos(s[sp]);
        break;
      case cTan:
        s[sp] = tan(s[sp]);
        break;
      case cSinh:
        s[sp] = sinh(s[sp]);
        break;
      case cCosh:
        s[sp] = cosh(s[sp]);
        break;
      case cTanh:
        s[sp] = tanh(s[sp]);
        break;
      case cCsc:
        s[sp] = 1.0 / sin(s[sp]);
        break;
      case cSec:
        s[sp] = 1.0 / cos(s[sp]);
        break;
      case cCot:
        s[sp] = 1.0 / tan(s[sp]);
        break;
      case cSinCos:
        s[sp + 1] = cos(s[sp]);
        s[sp] = sin(s[sp]);
        ++sp;
        break;
      case cSinhCosh:
        s[sp + 1] = cosh(s[sp]);
        s[sp] = sinh(s[sp]);
        ++sp;
        break;
      case cAsin:
        s[sp] = asin(s[sp]);
        break;
      case cAcos:
        s[sp] = acos(s[sp]);
        break;
      case cAsinh:
        s[sp] = asinh(s[sp]);
        break;
      case cAcosh:
        s[sp] = acosh(s[sp]);
        break;
      case cAtan:
        s[sp] = atan(s[sp]);
        break;
      case cAtanh:
        s[sp] = atanh(s[sp]);
        break;
      case cAtan2:
        --sp;
        s[sp] = atan2(s[sp], s[sp + 1]);
        break;
      case cHypot:
        --sp;
        s[sp] = torch::hypot(s[sp], s[sp + 1]);
        break;

      case cAbs:
        s[sp] = abs(s[sp]);
        break;
      case cMax:
        --sp;
        s[sp] = torch::maximum(s[sp], s[sp + 1]);
        break;
      case cMin:
        --sp;
        s[sp] = torch::minimum(s[sp], s[sp + 1]);
        break;
      case cTrunc:
        s[sp] = torch::trunc(s[sp]);
        break;
      case cCeil:
        s[sp] = torch::ceil(s[sp]);
        break;
      case cFloor:
        s[sp] = torch::floor(s[sp]);
        break;
      case cInt:
        s[sp] = torch::round(s[sp]);
        break;

        // case cEqual:
        //   //--sp; s[sp] = s[sp] == s[sp+1]; break;
        //   --sp;
        //   s[sp] = abs(s[sp] - s[sp + 1]) <= eps;
        //   break;
        // case cNEqual:
        //   //--sp; s[sp] = s[sp] != s[sp+1]; break;
        //   --sp;
        //   s[sp] = abs(s[sp] - s[sp + 1]) > eps;
        //   break;
        // case cLess:
        //   --sp;
        //   s[sp] = s[sp] < (s[sp + 1] - eps);
        //   break;
        // case cLessOrEq:
        //   --sp;
        //   s[sp] = s[sp] <= (s[sp + 1] + eps);
        //   break;
        // case cGreater:
        //   --sp;
        //   s[sp] = (s[sp] - eps) > s[sp + 1];
        //   break;
        // case cGreaterOrEq:
        //   --sp;
        //   s[sp] = (s[sp] + eps) >= s[sp + 1];
        //   break;
        // case cNot:
        //   s[sp] = abs(s[sp]) < 0.5;
        //   break;
        // case cNotNot:
        //   s[sp] = abs(s[sp]) >= 0.5;
        //   break;
        // case cAbsNot:
        //   s[sp] = s[sp] < 0.5;
        //   break;
        // case cAbsNotNot:
        //   s[sp] = s[sp] >= 0.5;
        //   break;
        // case cOr:
        //   --sp;
        //   s[sp] = (abs(s[sp]) >= 0.5) || (abs(s[sp + 1]) >= 0.5);
        //   break;
        // case cAbsOr:
        //   --sp;
        //   s[sp] = (s[sp] >= 0.5) || (s[sp + 1] >= 0.5);
        //   break;
        // case cAnd:
        //   --sp;
        //   s[sp] = (abs(s[sp]) >= 0.5) && (abs(s[sp + 1]) >= 0.5);
        //   break;
        // case cAbsAnd:
        //   --sp;
        //   s[sp] = (s[sp] >= 0.5) && (s[sp + 1] >= 0.5);
        //   break;

      case cLog:
        s[sp] = torch::log(s[sp]);
        break;
      case cLog2:
#ifdef FP_SUPPORT_CPLUSPLUS11_MATH_FUNCS
        s[sp - 1] = torch::log2(s[sp - 1]);
#else
        s[sp] = torch::log(s[sp]) / log(2.0);
#endif
        break;
      case cLog10:
        s[sp] = torch::log10(s[sp]);
        break;

      case cNeg:
        s[sp] = -s[sp];
        break;
      case cInv:
        s[sp] = 1.0 / s[sp];
        break;
      case cDeg:
        s[sp] = s[sp] * (180.0 / libMesh::pi);
        break;
      case cRad:
        s[sp] = s[sp] / (180.0 / libMesh::pi);
        break;

      case cFetch:
        ++sp;
        s[sp] = s[ByteCode[++i]];
        break;
      case cDup:
        ++sp;
        s[sp] = s[sp - 1];
        break;

      case cFCall:
      {
        auto function = ByteCode[++i];
        if (function == _mFFT)
        {
          if (s[sp].dim() == 1)
            s[sp] = torch::fft::rfft(s[sp]);
          else if (s[sp].dim() == 2)
            s[sp] = torch::fft::rfft2(s[sp]);
          else
            throw std::domain_error("3D not implemented yet");
        }
        else if (function == _miFFT)
        {
          if (s[sp].dim() == 1)
            s[sp] = torch::fft::irfft(s[sp]);
          else if (s[sp].dim() == 2)
            s[sp] = torch::fft::irfft2(s[sp]);
          else
            throw std::domain_error("3D not implemented yet");
        }
        else
          throw std::runtime_error("Function call not supported for libtorch tensors.");
      }
      break;

#ifdef FP_SUPPORT_OPTIMIZER
      case cPopNMov:
      {
        int dst = ByteCode[++i], src = ByteCode[++i];
        s[dst] = s[src];
        sp = dst;
        break;
      }
      case cLog2by:
        --sp;
        s[sp] = (torch::log(s[sp]) / std::log(2.0)) * s[sp + 1];
        break;
      case cNop:
        break;
#endif

      case cSqr:
        s[sp] = s[sp] * s[sp];
        break;
      case cSqrt:
        s[sp] = torch::sqrt(s[sp]);
        break;
      case cRSqrt:
        s[sp] = torch::pow(s[sp], -0.5);
        break;
      case cPow:
        --sp;
        s[sp] = torch::pow(s[sp], s[sp + 1]);
        break;
      case cExp:
        s[sp] = torch::exp(s[sp]);
        break;
      case cExp2:
        s[sp] = torch::pow(2.0, s[sp]);
        break;
      case cCbrt:
        s[sp] = torch::pow(s[sp], 1.0 / 3.0);
        break;

      case cJump:
      case cIf:
      case cAbsIf:
      {
        throw std::domain_error("Conditionals not implemented yet");
        // unsigned long ip = ByteCode[++i] + 1;

        // if (op == cIf)
        //   ccout << "if (abs(s[sp--]) < 0.5);
        // if (op == cAbsIf)
        //   ccout << "if (s[" << sp-- << "] < 0.5) ";

        // if (ip >= ByteCode.size())
        //   ccout << "*ret = s[sp]; return;
        // else
        // {
        //   ccout << "goto l" << ip << ";
        //   stackAtTarget[ip] = sp;
        // }

        // ++i;
        // break;
      }

      default:
        if (op >= VarBegin)
        {
          // load variable
          ++sp;
          s[sp] = *params[op - VarBegin];
        }
        else
        {
          throw std::runtime_error("Opcode not supported for libtorch tensors.");
        }
    }
  }

  return s[sp];
}
