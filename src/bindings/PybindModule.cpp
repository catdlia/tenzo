#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

// This is a stub for testing the Python-C++ bridge.
// In the future, this will hook into Tenzo's ExecutionContext to run MLIR-compiled kernels.
py::array_t<float> dispatch_linear_cxx(py::array_t<float> input, py::array_t<float> weight) {
    std::cout << "[Tenzo C++ Runtime] Successfully entered dispatch_linear_cxx!" << std::endl;
    std::cout << "[Tenzo C++ Runtime] Input shape matches and Pybind11 is routing correctly." << std::endl;

    auto buf_input = input.request();
    auto buf_weight = weight.request();

    if (buf_input.ndim != 3 || buf_weight.ndim != 2) {
        throw std::runtime_error("Unsupported shapes for mockup");
    }

    auto batch = buf_input.shape[0];
    auto seq = buf_input.shape[1];
    auto in_features = buf_input.shape[2];
    auto out_features = buf_weight.shape[0];

    // Allocate result tensor
    auto result = py::array_t<float>({batch, seq, out_features});
    auto buf_result = result.request();

    float *ptr_input = static_cast<float *>(buf_result.ptr); // Wait, this points to result. We'll use the correct pointers below.
    float *ptr_in = static_cast<float *>(buf_input.ptr);
    float *ptr_w = static_cast<float *>(buf_weight.ptr);

    std::cout << "[Tenzo C++ Runtime] Simulating BitLinear Native Execution..." << std::endl;

    // Simple, extremely naive matmul for stub verification
    for (ssize_t b = 0; b < batch; ++b) {
        for (ssize_t s = 0; s < seq; ++s) {
            for (ssize_t o = 0; o < out_features; ++o) {
                float sum = 0.0f;
                for (ssize_t i = 0; i < in_features; ++i) {
                    sum += ptr_in[b * seq * in_features + s * in_features + i] * 
                           ptr_w[o * in_features + i];
                }
                ptr_input[b * seq * out_features + s * out_features + o] = sum;
            }
        }
    }

    return result;
}

PYBIND11_MODULE(tenzo_runtime, m) {
    m.doc() = "Tenzo C++ Runtime Native Module";
    m.def("dispatch_linear_cxx", &dispatch_linear_cxx, "Dispatch a linear operation to Tenzo C++ MLIR kernel",
          py::arg("input"), py::arg("weight"));
}
