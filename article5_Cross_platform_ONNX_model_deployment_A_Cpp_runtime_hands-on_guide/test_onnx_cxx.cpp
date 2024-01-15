#include <iostream>
#include <vector>
#include <queue>
#include <cpu_provider_factory.h> // For ONNXRuntime CPU provider
#include <onnxruntime_cxx_api.h>  // OnnxRuntime C++ API

// Function to get the top K indices of the highest values in an array.
std::vector<int> getTopKIndices(const float* arr, int n, int k) {
    // Custom comparator for sorting in descending order based on the first element of the pair.
    auto comp = [&](const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first < b.first;
    };

    // Priority queue to store the pairs of value and index, sorted by the value.
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(comp)> pq(comp);

    // Pushing each element of the array along with its index into the priority queue.
    for (int i = 0; i < n; i++) {
        pq.push({arr[i], i});
    }

    // Vector to store the indices of the top K elements.
    std::vector<int> topKIndices;
    // Extracting the top K elements.
    for (int i = 0; i < k && !pq.empty(); i++) {
        topKIndices.push_back(pq.top().second); // Pushing the index of the top element
        pq.pop(); // Removing the top element from the queue
    }
    return topKIndices;
}

int main() {
    // Initialize ONNX Runtime environment.
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TestONNXRuntime");
    Ort::SessionOptions session_options;
    // Load the ONNX model.
    Ort::Session session(env, "/model/lstm_model.onnx", session_options);

    // Define the input data for the model.
    std::vector<float> seq_input_data = {1.0f, 2.0f, 3.0f, 9.0f, 5.0f, 6.0f}; // Changed to float
    std::vector<int64_t> seq_input_shape = {1, 6, 1};

    // Memory information for creating the input tensor.
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    // Create input tensor.
    Ort::Value seq_input_tensor = Ort::Value::CreateTensor<float>(memory_info, seq_input_data.data(), seq_input_data.size(), seq_input_shape.data(), seq_input_shape.size());

    // Define the names of the input tensors.
    std::vector<const char*> input_names = {"seq_input"};
    // Vector to store the input tensors.
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(seq_input_tensor));

    // Define the names of the output tensors.
    const char* output_names[] = {"my_output"};
    // Run the model.
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(), output_names, 1);

    // Extract the output tensor data.
    auto* floatarr = output_tensors[0].GetTensorMutableData<float>();
    // Get the shape of the output tensor.
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    // Calculate the total number of elements in the output tensor.
    int total_elements = 1;
    for (int dim : output_shape) {
        total_elements *= dim;
    }

    // Get the top indices from the output tensor.
    auto topIndices = getTopKIndices(floatarr, total_elements, 1);

    // Print the top indices.
    std::cout << "Top Indices: ";
    for (int index : topIndices) {
        std::cout << index << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
