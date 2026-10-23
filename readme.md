# ONNX RuntimeC++ static library installation, download and usage tutorial

### Differences between Static and Dynamic Library

#### Static Library

- **File Extension**

: `.lib` (Windows), `.a` (Linux/macOS)

- **Linking method**

Compile-time linking, the code is copied to the final executable file.

- **Deployment**

: A single executable file, no need for additional DLLs.

- **Size**

: The executable file is large.

- **Compatibility**

: Better, not dependent on system environment

#### Dynamic Library

- **File Extension**

: `.dll` (Windows), `.so` (Linux), `.dylib` (macOS)

- **Linking method**

: Loaded at runtime, multiple programs can share it.

- **Deployment**

: Requires the distribution of DLL files

- **Size**

: Executable file is small.

- **Memory usage**

: More efficient (shared library)

ONNX Runtime static library usage tutorial

#### 1. Environmental Preparation

#### Download ONNX Runtime

Download the precompiled package from official GitHub or compile it yourself.

Directory Structure:

```
onnxruntime/
├── include/
├── lib/
├── onnxruntime.lib (static library)
└── onnxruntime.dll (dynamic library)
└── bin/
```

#### 2. CMake Configuration Example

```
cmake_minimum_required(VERSION 3.18)
project(ONNXRuntimeDemo)
set(CMAKE_CXX_STANDARD 14)
# Set ONNX Runtime Path
set(ONNXRUNTIME_ROOT "path/to/onnxruntime")
# Contains a table of contents
include_directories(${ONNXRUNTIME_ROOT}/include)
# Static Library Linking
if(WIN32)
# Windows Static Library Configuration
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
# Required system libraries
    set(SYSTEM_LIBS 
        ws2_32
        advapi32
        dbghelp
    )
else()
# Linux/macOS static library configuration
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.a)
# Required system libraries
    set(SYSTEM_LIBS
        pthread
        dl
    )
endif()
# Creating an executable file
add_executable(onnx_demo main.cpp)
# Linked Library
target_link_libraries(onnx_demo 
    ${ONNXRUNTIME_LIB}
    ${SYSTEM_LIBS}
)
```

#### 3. C++ Code Example

```
[#include] <onnxruntime_cxx_api.h>
[#include] <iostream>
[#include] <vector>
[#include] <chrono>
class ONNXModel {
private:
    Ort::Env env;
    Ort::Session session;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
public:
    ONNXModel(const std::string& model_path, int device_id = 0) 
        : env(ORT_LOGGING_LEVEL_WARNING, "ONNXModel") {
// Session options
        Ort::SessionOptions session_options;
// Set the number of threads
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);
// For GPU usage, uncomment the following lines.
        // OrtCUDAProviderOptions cuda_options;
        // cuda_options.device_id = device_id;
        // session_options.AppendExecutionProvider_CUDA(cuda_options);
// Create a session
        session = Ort::Session(env, model_path.c_str(), session_options);
// Acquire input and output information
        setup_io_names();
    }
private:
    void setup_io_names() {
        Ort::AllocatorWithDefaultOptions allocator;
// Input name
        size_t num_input_nodes = session.GetInputCount();
        for(size_t i = 0; i < num_input_nodes; i++) {
            char* input_name = session.GetInputName(i, allocator);
            input_names.push_back(input_name);
        }
// Output Name
        size_t num_output_nodes = session.GetOutputCount();
        for(size_t i = 0; i < num_output_nodes; i++) {
            char* output_name = session.GetOutputName(i, allocator);
            output_names.push_back(output_name);
        }
    }
public:
    std::vector<float> inference(const std::vector<float>& input_data, 
                                const std::vector<int64_t>& input_shape) {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
// Create input tensor
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, 
            const_cast<float*>(input_data.data()), 
            input_data.size(),
            input_shape.data(), 
            input_shape.size()
        );
// Run the inference
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, 
            input_names.data(), 
            &input_tensor, 
            input_names.size(),
            output_names.data(), 
            output_names.size()
        );
// Get the output
        float* floatarr = output_tensors[0].GetTensorMutableData<float>();
        auto tensor_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t output_size = 1;
        for(auto dim : tensor_shape) {
            output_size *= dim;
        }
        return std::vector<float>(floatarr, floatarr + output_size);
    }
    void print_model_info() {
        Ort::AllocatorWithDefaultOptions allocator;
        std::cout << "=== Model Information ===" << std::endl;
// Input information
        std::cout << "Inputs:" << std::endl;
        size_t num_inputs = session.GetInputCount();
        for(size_t i = 0; i < num_inputs; i++) {
            auto type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            char* name = session.GetInputName(i, allocator);
            std::cout << "  " << i << ": " << name << " Shape: [";
            for(size_t j = 0; j < shape.size(); j++) {
                std::cout << shape[j];
                if(j < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
// Output information
        std::cout << "Outputs:" << std::endl;
        size_t num_outputs = session.GetOutputCount();
        for(size_t i = 0; i < num_outputs; i++) {
            auto type_info = session.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            char* name = session.GetOutputName(i, allocator);
            std::cout << "  " << i << ": " << name << " Shape: [";
            for(size_t j = 0; j < shape.size(); j++) {
                std::cout << shape[j];
                if(j < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
};
// Example usage
int main() {
    try {
// Initialize the model
        ONNXModel model("model.onnx");
// Print model information
        model.print_model_info();
// Preparing input data (example)
std::vector<float> input_data(1 * 3 * 224 * 224, 0.5f); // Assuming it is an image classification model
        std::vector<int64_t> input_shape = {1, 3, 224, 224};
// Execute inference
        auto start_time = std::chrono::high_resolution_clock::now();
        auto output = model.inference(input_data, input_shape);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
// Output result
        std::cout << "Output size: " << output.size() << std::endl;
        std::cout << "First 10 values: ";
        for(int i = 0; i < 10 && i < output.size(); i++) {
            std::cout << output[i] << " ";
        }
        std::cout << std::endl;
    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
```

4. Compilation and Execution

##### Windows (command line)

```
mkdir build
cd build
cmake -G "Visual Studio 16 2019" ..
cmake --build . --config Release
```

##### Linux/macOS

```
mkdir build
cd build
cmake ..
make -j4
```

5. Advanced Features

##### Using GPU

```
// Add GPU support in the constructor.
[#ifdef] USE_CUDA
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = device_id;
    session_options.AppendExecutionProvider_CUDA(cuda_options);
[#endif]
```

##### Custom Memory Allocation

```
class CustomAllocator : public Ort::Allocator {
// Implement a custom memory allocator
};
```

### Notes

1. **Static Library Size**

: ONNX Runtime static library is large, consider using dynamic library if you are sensitive to the size of the executable file.

1. **Dependency Management**

: Static library contains all dependencies, ensuring no conflicting system library versions.

1. **Compilation Time**

: Static linking may increase compilation time.

1. **Memory usage**

: Each program that uses a static library has its own separate ONNX Runtime copy.

### Common Issues

1. **Link Error**

Ensure all necessary system libraries are linked.

1. **Version compatibility**

Ensure that the ONNX Runtime version is compatible with the model.

1. **Memory Leakage**

: Manage Ort objects using RAII

This tutorial provides the basic usage of ONNX Runtime static library, which you can adjust and extend according to specific needs. Regarding the ONNX Runtime static library, it is currently available for download from the GitHub repository onnxruntime_chinese_mirror, with the following version:

|

Project Name

 |
|

[C++onnxruntime static library] onnxruntime-win-x64-static-lib-1.23.1

 |
|

[C++onnxruntime static library] onnxruntime-win-x64-static-lib-1.22.1

 |
|

[C++onnxruntime static library] onnxruntime-win-x64-static-lib-1.22.0

 |
|

[C++onnxruntime static library] onnxruntime-win-x64-static-lib-1.20.1

 |
|

[C++onnxruntime static library] onnxruntime-win-x64-static-lib-1.20.0

 |
|

[C++onnxruntime static library] onnxruntime-win-x64-static-lib-1.19.2

 |
|

[C++onnxruntime static library] onnxruntime-win-x64-static-lib-1.19.0

 |
|

[C++onnxruntime static library] onnxruntime-win-x64-static-lib-1.18.1

 |
|

[C++onnxruntime static library] onnxruntime-win-x64-static-lib-1.18.0

 |
|

[C++onnxruntime static library] onnxruntime-win-x64-static-lib-1.17.3

 |
|

[C++onnxruntime static library] onnxruntime-win-x64-static-lib-1.17.1

 |
|

[C++onnxruntime static library] onnxruntime-win-x64-static-lib-1.17.0

 |
|

[C++onnxruntime static library] onnxruntime-win-x64-static-lib-1.16.3

 |
|

[C++onnxruntime static library] onnxruntime-win-x64-static-lib-1.16.2

 |
|

[C++onnxruntime static library] onnxruntime-win-x64-static-lib-1.16.1

 |
|

[C++onnxruntime static library] onnxruntime-win-x64-static-lib-1.16.0

 |
|

[C++onnxruntime static library] onnxruntime-win-x86-static-lib-1.23.1

 |
|

[C++onnxruntime static library] onnxruntime-win-x86-static-lib-1.22.1

 |
|

[C++onnxruntime static library] onnxruntime-win-x86-static-lib-1.22.0

 |
|

[C++onnxruntime static library] onnxruntime-win-x86-static-lib-1.20.1

 |
|

[C++onnxruntime static library] onnxruntime-win-x86-static-lib-1.19.2

 |
|

[C++onnxruntime static library] onnxruntime-win-x86-static-lib-1.19.0

 |
|

[C++onnxruntime static library] onnxruntime-win-x86-static-lib-1.18.1

 |
|

[C++onnxruntime static library] onnxruntime-win-x86-static-lib-1.18.0

 |
|

[C++onnxruntime static library] onnxruntime-win-x86-static-lib-1.17.3

 |
|

[C++onnxruntime static library] onnxruntime-win-x86-static-lib-1.17.1

 |
|

[C++onnxruntime static library] onnxruntime-win-x86-static-lib-1.17.0

 |
|

[C++onnxruntime static library] onnxruntime-win-x86-static-lib-1.16.3

 |
|

[C++onnxruntime static library] onnxruntime-win-x86-static-lib-1.16.2

 |
|

[C++onnxruntime static library] onnxruntime-win-x86-static-lib-1.16.1

 |
|

[C++onnxruntime static library] onnxruntime-win-x86-static-lib-1.16.0

 |


## Payment

Here is a pay link on Stripe ( https://buy.stripe.com/3cs8yP7sY87d0vu9AB ). Please contact me lonlonago@foxmail.com after funding $89, and I will send you a complete data files , thank you!

![111.png](111.png)
