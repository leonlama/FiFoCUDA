// Logic to load and prepare TensorRT engine
#include "engine_loader.h"
#include <NvInfer.h>
#include <fstream>
#include <iostream>

IExecutionContext* loadEngine(const char* enginePath) {
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (!engineFile) {
        std::cerr << "Error opening engine file: " << enginePath << std::endl;
        return nullptr;
    }

    // Get engine file size
    engineFile.seekg(0, engineFile.end);
    size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    // Read engine data
    char* engineData = new char[engineSize];
    engineFile.read(engineData, engineSize);
    engineFile.close();

    // Create runtime and deserialize engine
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData, engineSize, nullptr);

    delete[] engineData;
    if (!engine) {
        std::cerr << "Failed to deserialize engine." << std::endl;
        return nullptr;
    }

    // Create execution context
    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context." << std::endl;
        engine->destroy();
    }

    return context;
}
