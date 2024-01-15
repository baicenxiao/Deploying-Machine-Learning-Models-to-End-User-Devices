#include <onnxruntime_training_cxx_api.h>
#include <vector>
#include <iostream>

// Structure to hold training data
struct TrainingData {
    std::vector<std::vector<std::vector<float>>> x; // 3D vector to hold reshaped sequences
    // std::vector<int> y;
    std::vector<int64_t> y;
};

// Function to generate training data
TrainingData generateTrainingData(const std::vector<int>& fullSequence, int sequenceLength = 16) {
    std::vector<std::vector<std::vector<float>>> x;
    std::vector<int64_t> y;

    for (size_t i = 1; i <= fullSequence.size(); ++i) {
        std::vector<std::vector<float>> subsequence(sequenceLength, std::vector<float>(1, 0)); // Initialize with zeros
        for (size_t j = 0; j < i; ++j) {
            subsequence[sequenceLength - i + j][0] = fullSequence[j]; // Assign sequence values
        }
        int label = (i < fullSequence.size()) ? fullSequence[i] : fullSequence[0];

        x.push_back(subsequence);
        y.push_back(label);
    }

    return {x, y};
}


std::vector<std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<int64_t>>> createBatches(const std::vector<std::vector<std::vector<float>>>& features, const std::vector<int64_t>& labels, int batchSize) {
    std::vector<std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<int64_t>>> batches;

    for (size_t start = 0; start < features.size(); start += batchSize) {
        size_t end = std::min(start + batchSize, features.size());
        std::vector<std::vector<std::vector<float>>> batchFeatures(features.begin() + start, features.begin() + end);
        std::vector<int64_t> batchLabels(labels.begin() + start, labels.begin() + end);

        batches.emplace_back(batchFeatures, batchLabels);
    }

    return batches;
}


float trainStep(const std::vector<std::vector<std::vector<float>>>& inputData, const std::vector<int64_t>& labels, Ort::TrainingSession& trainingSession) {    // Flatten the inputData
    std::vector<float> flatInputData;
    for (const auto& sequence : inputData) {
        for (const auto& data : sequence) {
            flatInputData.insert(flatInputData.end(), data.begin(), data.end());
        }
    }

    // Create a non-const copy of labels data
    std::vector<int64_t> nonConstLabels(labels.begin(), labels.end());

    // Define the shape for input
    const std::vector<int64_t> inputShape = {static_cast<int64_t>(inputData.size()), 16, 1};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> userInputs;

    // Create tensor for input data
    userInputs.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, flatInputData.data(),
                                                            flatInputData.size(),
                                                            inputShape.data(), inputShape.size()));

    // Define the shape for labels
    const std::vector<int64_t> labelsShape = {static_cast<int64_t>(labels.size())};


    // Create tensor for labels using non-const copy
    userInputs.emplace_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, nonConstLabels.data(),
                                                              nonConstLabels.size(),
                                                              labelsShape.data(), labelsShape.size()));


    // Run the train step
    float loss = *(trainingSession.TrainStep(userInputs).front().GetTensorMutableData<float>());

    // Update model parameters
    trainingSession.OptimizerStep();

    // Reset gradients
    trainingSession.LazyResetGrad();

    return loss;
}

int main() {
    Ort::Env env;
    Ort::SessionOptions session_options;

    auto state = Ort::CheckpointState::LoadCheckpoint("./training_artifacts/checkpoint");
    auto training_session = Ort::TrainingSession(env, session_options, state, "./training_artifacts/training_model.onnx", "./training_artifacts/eval_model.onnx", "./training_artifacts/optimizer_model.onnx");

    // Generate training data
    std::vector<int> fullSequence = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    int sequenceLength = 16;
    auto [xTrain, yTrain] = generateTrainingData(fullSequence, sequenceLength);

    // Create batches
    int batchSize = 16;
    auto batches = createBatches(xTrain, yTrain, batchSize); // yTrain is int64_t

    int numEpochs = 100;

    // Training loop
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        std::cout << "Epoch " << (epoch + 1) << "/" << numEpochs << std::endl;
        float loss;
        for (auto& batch : batches) {
            auto& [batchInput, batchLabels] = batch;
            loss = trainStep(batchInput, batchLabels, training_session);
            // Print loss value for each batch
            std::cout << "in batch" << std::endl;
        }
        std::cout << "Epoch Loss: " << loss << std::endl;

        // Additional code for model evaluation and checkpoint saving can be added here
    }

    std::string inference_model_path = "./inference_model.onnx";
    std::vector<std::string> output_node_names = {"output"}; // Replace with your actual output node names

    try {
        training_session.ExportModelForInferencing(inference_model_path, output_node_names);
        std::cout << "Model exported successfully to: " << inference_model_path << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "Error exporting model: " << e.what() << std::endl;
    }



    std::string path_to_checkpoint = "./training_checkpoint.chkpt";
    Ort::CheckpointState::SaveCheckpoint(state, path_to_checkpoint, false);


    return 0;
}
