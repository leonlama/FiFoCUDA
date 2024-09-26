#include "inference.h"
#include <cuda_runtime.h>
#include <iostream>

// Allocate memory for the input tensor in GPU VRAM
float* allocateInputTensor(int channels, int height, int width, int batchSize) {
    float* d_input;  // Pointer to the input tensor in device memory (GPU)
    size_t tensorSize = channels * height * width * batchSize * sizeof(float);  // Calculate the tensor size

    // Allocate memory on the GPU for the input tensor
    cudaError_t err = cudaMalloc(&d_input, tensorSize);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for input tensor: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }

    // Optionally initialize the tensor memory to zeros (or any other initialization)
    cudaMemset(d_input, 0, tensorSize);

    std::cout << "Allocated input tensor of size " << tensorSize << " bytes in GPU memory." << std::endl;
    return d_input;
}

// Perform inference on the GPU using TensorRT
void runInference(IExecutionContext* context, float* inputTensor, int batchSize) {
    // Create a CUDA stream for asynchronous execution
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate memory for the output tensor (output size depends on your model)
    // Assuming you know the size of the output tensor (this will vary based on your model)
    float* d_output;
    size_t outputSize = /* Define output size based on the model's output */;
    cudaMalloc(&d_output, outputSize);

    // Set up the input and output buffers (binding tensors to the engine)
    void* buffers[] = { inputTensor, d_output };

    // Enqueue inference in TensorRT's execution context
    if (!context->enqueue(batchSize, buffers, stream, nullptr)) {
        std::cerr << "Failed to run inference!" << std::endl;
    }

    // Synchronize the CUDA stream to wait for the inference to complete
    cudaStreamSynchronize(stream);

    std::cout << "Inference completed successfully." << std::endl;

    // Clean up GPU memory allocated for the output tensor
    cudaFree(d_output);
    cudaStreamDestroy(stream);
}

// Free the memory allocated for the input tensor
void freeInputTensor(float* tensor) {
    cudaFree(tensor);
}
