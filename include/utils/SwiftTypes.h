#pragma once

#include <string>

/// Name of an TensorOperator object
// DerivativeStringClass(TensorComputeName);
typedef std::string TensorComputeName;

/// Name of an TensorPredictor object
// DerivativeStringClass(TensorPredictorName);
typedef std::string TensorPredictorName;

/// Name of an TensorTimeIntegrator object
// DerivativeStringClass(TensorTimeIntegratorName);
typedef std::string TensorTimeIntegratorName;

/// Name of a read-only input TensorBuffer object
// DerivativeStringClass(TensorInputBufferName);
typedef std::string TensorInputBufferName;
/// Name of a writable output TensorBuffer object
// DerivativeStringClass(TensorOutputBufferName);
typedef std::string TensorOutputBufferName;

/// Name of an TensorOutput object
// DerivativeStringClass(TensorOutputName);
typedef std::string TensorOutputName;

/// Forward declarations
class TensorBufferBase;
class TensorOperator;
