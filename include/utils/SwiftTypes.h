#pragma once

#include <string>

/// Name of an TensorOperator object
// DerivativeStringClass(FFTComputeName);
typedef std::string FFTComputeName;

/// Name of an FFTIC object
// DerivativeStringClass(FFTICName);
typedef std::string FFTICName;

/// Name of an TensorTimeIntegrator object
// DerivativeStringClass(FFTTimeIntegratorName);
typedef std::string FFTTimeIntegratorName;

/// Name of a read-only input TensorBuffer object
// DerivativeStringClass(FFTInputBufferName);
typedef std::string FFTInputBufferName;
/// Name of a writable output TensorBuffer object
// DerivativeStringClass(FFTOutputBufferName);
typedef std::string FFTOutputBufferName;

/// Name of an TensorOutput object
// DerivativeStringClass(FFTOutputName);
typedef std::string FFTOutputName;

/// Forward declarations
class TensorBufferBase;
class TensorOperator;
