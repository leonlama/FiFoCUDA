// TensorRT engine loading functions
#ifndef ENGINE_LOADER_H
#define ENGINE_LOADER_H

#include <NvInfer.h>  // TensorRT headers

// Function to load a TensorRT engine from the .engine file and create an execution context
nvinfer1::IExecutionContext* loadEngine(const char* enginePath);

#endif  // ENGINE_LOADER_H
