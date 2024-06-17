#pragma once

#include <string>

/// Name of an FFTCompute object
// DerivativeStringClass(FFTComputeName);
typedef std::string FFTComputeName;

/// Name of an FFTIC object
// DerivativeStringClass(FFTICName);
typedef std::string FFTICName;

/// Name of a read-only input FFTBuffer object
// DerivativeStringClass(FFTInputBufferName);
typedef std::string FFTInputBufferName;
/// Name of a writable output FFTBuffer object
// DerivativeStringClass(FFTOutputBufferName);
typedef std::string FFTOutputBufferName;

/// Forward declarations
class FFTBufferBase;
class FFTCompute;
