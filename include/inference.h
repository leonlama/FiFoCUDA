// CUDA inference function declarations
#ifndef INFERENCE_H
#define INFERENCE_H

#include <NvInfer.h>  // TensorRT header for IExecutionContext

// Function to allocate input tensor memory in GPU VRAM
float* allocateInputTensor(int channels, int height, int width, int batchSize);

// Function to perform inference using the loaded TensorRT engine
void runInference(nvinfer1::IExecutionContext* context, float* inputTensor, int batchSize);

// Function to free the memory allocated for the input tensor
void freeInputTensor(float* tensor);

#endif  // INFERENCE_H
