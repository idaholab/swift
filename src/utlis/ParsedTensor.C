//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#pragma once

#include "ParsedTensor.h"
#include "libmesh/fptypes.hh"

void
ParsedTensor::setupTensors()
{
  // allocate stack
  s.resize(getParserData()->mStackSize);

  // convert immediate data
  tensor_immed.clear();
  for (const auto & i : getParserData()->mImmed)
    tensor_immed.push_back(neml2::Scalar(i, neml2::default_tensor_options()));
}

neml2::Scalar
ParsedTensor::CustomEval(const neml2::Scalar * Vars)
{
  // get a reference to the stored bytecode
  const auto & ByteCode = getParserData()->mByteCode;

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
        s[sp] += s[sp + 1];
        break;
      case cSub:
        --sp;
        s[sp] -= s[sp + 1];
        break;
      case cRSub:
        --sp;
        s[sp] = s[sp + 1] - s[sp];
        break;
      case cMul:
        --sp;
        s[sp] *= s[sp + 1];
        break;
      case cDiv:
        --sp;
        s[sp] /= s[sp + 1];
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
        s[sp] = sqrt(s[sp] * s[sp] + s[sp + 1] * s[sp + 1]);
        break;

      case cAbs:
        s[sp] = abs(s[sp]);
        break;
      case cMax:
        --sp;
        s[sp] = s[sp] > s[sp + 1] ? s[sp] : s[sp + 1];
        break;
      case cMin:
        --sp;
        s[sp] = s[sp] < s[sp + 1] ? s[sp] : s[sp + 1];
        break;
      case cTrunc:
        s[sp] = s[sp] < 0 ? ceil(s[sp]) : floor(s[sp]);
        break;
      case cCeil:
        s[sp] = ceil(s[sp]);
        break;
      case cFloor:
        s[sp] = floor(s[sp]);
        break;
      case cInt:
        s[sp] = s[sp] < 0 ? ceil(s[sp] - 0.5) : floor(s[sp] + 0.5);
        break;

      case cEqual:
        //--sp; s[sp] = s[sp] == s[sp+1]; break;
        --sp;
        s[sp] = abs(s[sp] - s[sp + 1]) <= eps;
        break;
      case cNEqual:
        //--sp; s[sp] = s[sp] != s[sp+1]; break;
        --sp;
        s[sp] = abs(s[sp] - s[sp + 1]) > eps;
        break;
      case cLess:
        --sp;
        s[sp] = s[sp] < (s[sp + 1] - eps);
        break;
      case cLessOrEq:
        --sp;
        s[sp] = s[sp] <= (s[sp + 1] + eps);
        break;
      case cGreater:
        --sp;
        s[sp] = (s[sp] - eps) > s[sp + 1];
        break;
      case cGreaterOrEq:
        --sp;
        s[sp] = (s[sp] + eps) >= s[sp + 1];
        break;
      case cNot:
        s[sp] = abs(s[sp]) < 0.5;
        break;
      case cNotNot:
        s[sp] = abs(s[sp]) >= 0.5;
        break;
      case cAbsNot:
        s[sp] = s[sp] < 0.5;
        break;
      case cAbsNotNot:
        s[sp] = s[sp] >= 0.5;
        break;
      case cOr:
        --sp;
        s[sp] = (abs(s[sp]) >= 0.5) || (abs(s[sp + 1]) >= 0.5);
        break;
      case cAbsOr:
        --sp;
        s[sp] = (s[sp] >= 0.5) || (s[sp + 1] >= 0.5);
        break;
      case cAnd:
        --sp;
        s[sp] = (abs(s[sp]) >= 0.5) && (abs(s[sp + 1]) >= 0.5);
        break;
      case cAbsAnd:
        --sp;
        s[sp] = (s[sp] >= 0.5) && (s[sp + 1] >= 0.5);
        break;

      case cLog:
        s[sp] = log(s[sp]);
        break;
      case cLog2:
#ifdef FP_SUPPORT_CPLUSPLUS11_MATH_FUNCS
        s[sp - 1] = log2(s[sp - 1]);
#else
        s[sp] = log(s[sp]) / log(2.0);
#endif
        break;
      case cLog10:
        s[sp] = log10(s[sp]);
        break;

      case cNeg:
        s[sp] = -s[sp];
        break;
      case cInv:
        s[sp] = 1.0 / s[sp];
        break;
      case cDeg:
        s[sp] *= 180.0 / M_PI;
        break;
      case cRad:
        s[sp] /= 180.0 / M_PI;
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
        unsigned function = ByteCode[++i];
        if (function == mFPlog)
        {
          --sp;
          s[sp] = s[sp] < s[sp + 1] ? log(s[sp + 1]) + (s[sp] - s[sp + 1]) / s[sp + 1] -
                                          pow((s[sp] - s[sp + 1]) / s[sp + 1], 2.0) / 2.0 +
                                          pow((s[sp] - s[sp + 1]) / s[sp + 1], 3.0) / 3.0
                                    : log(s[sp]);
        }
        else if (function == mFErf)
        {
#if LIBMESH_HAVE_CXX11_ERF
          s[sp] = erf(s[sp]);
#else
          cerr << "Libmesh is not compiled with c++11 so erf is not supported by JIT.\n";
          return false;
#endif
        }
        else
        {
          cerr << "Function call not supported by JIT.\n";
          return false;
        }
        break;
      }

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
        s[sp] = log(s[sp]) / log(2.0) * s[sp + 1];
        break;
      case cNop:
        break;
#endif

      case cSqr:
        s[sp] *= s[sp];
        break;
      case cSqrt:
        s[sp] = sqrt(s[sp]);
        break;
      case cRSqrt:
        s[sp] = pow(s[sp], (-0.5));
        break;
      case cPow:
        --sp;
        s[sp] = pow(s[sp], s[sp + 1]);
        break;
      case cExp:
        s[sp] = exp(s[sp]);
        break;
      case cExp2:
        s[sp] = pow(2.0, s[sp]);
        break;
      case cCbrt:
#ifdef FP_SUPPORT_CPLUSPLUS11_MATH_FUNCS
        s[sp] = cbrt(s[sp]);
        break;
#else
        s[sp] = s[sp] == 0 ? 0 : (s[sp] > 0 ? exp(log(s[sp]) / 3.0) : -exp(log(-s[sp]) / 3.0));
        break;
#endif

      case cJump:
      case cIf:
      case cAbsIf:
      {
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
          s[sp] = params[op - VarBegin];
        }
        else
        {
          cerr << "Opcode not supported by JIT.\n";
          return false;
        }
    }
  }

  return s[sp];
}
