// Logic to load and prepare TensorRT engine
#include "engine_loader.h"
#include <NvInfer.h>
#include <fstream>
#include <iostream>

// Logger for TensorRT info/warnings/errors
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) override {
        if (severity != Severity::kINFO) {  // Ignore INFO-level logs
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

// Function to load the TensorRT engine from a file
IExecutionContext* loadEngine(const char* enginePath) {
    // Open the engine file in binary mode
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (!engineFile) {
        std::cerr << "Error: Could not open engine file: " << enginePath << std::endl;
        return nullptr;
    }

    // Get the size of the engine file
    engineFile.seekg(0, engineFile.end);
    size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    // Read the engine data into memory
    char* engineData = new char[engineSize];
    engineFile.read(engineData, engineSize);
    engineFile.close();

    // Create the TensorRT runtime
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Error: Failed to create TensorRT runtime." << std::endl;
        delete[] engineData;
        return nullptr;
    }

    // Deserialize the engine
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData, engineSize, nullptr);
    delete[] engineData;  // Clean up engine data buffer

    if (!engine) {
        std::cerr << "Error: Failed to deserialize TensorRT engine." << std::endl;
        return nullptr;
    }

    // Create the execution context from the engine
    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Error: Failed to create execution context." << std::endl;
        engine->destroy();
        return nullptr;
    }

    return context;  // Return the created execution context
}

