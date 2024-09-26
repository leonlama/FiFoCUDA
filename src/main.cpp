#include "inference.h"
#include "engine_loader.h"
#include <iostream>

int main() {
    const char* enginePath = "models/model.engine";

    // Load TensorRT engine
    IExecutionContext* context = loadEngine(enginePath);
    if (!context) {
        std::cerr << "Failed to load engine!" << std::endl;
        return -1;
    }

    // Define input dimensions (3x244x244x10)
    const int batchSize = 10;
    const int channels = 3;
    const int height = 244;
    const int width = 244;

    // Allocate and initialize input tensor
    float* inputTensor = allocateInputTensor(channels, height, width, batchSize);

    // Run inference
    runInference(context, inputTensor, batchSize);

    // Clean up and free resources
    freeInputTensor(inputTensor);
    context->destroy();

    return 0;
}
