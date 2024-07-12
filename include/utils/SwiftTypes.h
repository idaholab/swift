#pragma once

#include <string>

/// Name of an TensorOperator object
// DerivativeStringClass(FFTComputeName);
typedef std::string FFTComputeName;

/// Name of an FFTIC object
// DerivativeStringClass(FFTICName);
typedef std::string FFTICName;

/// Name of an FFTTimeIntegrator object
// DerivativeStringClass(FFTTimeIntegratorName);
typedef std::string FFTTimeIntegratorName;

/// Name of a read-only input FFTBuffer object
// DerivativeStringClass(FFTInputBufferName);
typedef std::string FFTInputBufferName;
/// Name of a writable output FFTBuffer object
// DerivativeStringClass(FFTOutputBufferName);
typedef std::string FFTOutputBufferName;

/// Name of an FFTOutput object
// DerivativeStringClass(FFTOutputName);
typedef std::string FFTOutputName;

/// Forward declarations
class FFTBufferBase;
class TensorOperator;
