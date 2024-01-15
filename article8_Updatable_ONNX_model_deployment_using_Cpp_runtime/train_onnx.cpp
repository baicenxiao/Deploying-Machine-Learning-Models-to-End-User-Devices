#include <onnxruntime_training_cxx_api.h>
#include <vector>
#include <iostream>
#include <random>

// Structure to hold training data

struct TrainingData {
    std::vector<std::vector<float>> x;
    std::vector<int64_t> y;
};

// Function to generate random float values
float randomFloat(float min, float max) {
    static std::random_device rd;
    static std::mt19937 e(rd());
    std::uniform_real_distribution<> dist(min, max);
    return dist(e);
}

// Function to generate training data
TrainingData generateTrainingData(int dataSize = 1000) {
    TrainingData data;
    data.x.reserve(dataSize);
    data.y.reserve(dataSize);

    for (int i = 0; i < dataSize; ++i) {
        std::vector<float> randomValues(6);
        float sum = 0.0f;
        for (float &value : randomValues) {
            value = randomFloat(0.0f, 20.0f);
            sum += value;
        }
        data.x.push_back(randomValues);
        data.y.push_back(static_cast<int64_t>(sum / 40));
    }

    return data;
}

// Function to create batches
std::vector<std::pair<std::vector<std::vector<float>>, std::vector<int64_t>>> createBatches(const TrainingData& data, int batchSize) {
    std::vector<std::pair<std::vector<std::vector<float>>, std::vector<int64_t>>> batches;

    for (size_t start = 0; start < data.x.size(); start += batchSize) {
        size_t end = std::min(start + batchSize, data.x.size());
        std::vector<std::vector<float>> batchFeatures(data.x.begin() + start, data.x.begin() + end);
        std::vector<int64_t> batchLabels(data.y.begin() + start, data.y.begin() + end);

        batches.emplace_back(batchFeatures, batchLabels);
    }

    return batches;
}


float trainStep(const std::vector<std::vector<float>>& inputData, const std::vector<int64_t>& labels, Ort::TrainingSession& trainingSession) {
    // Flatten the inputData
    std::vector<float> flatInputData;
    for (const auto& data : inputData) {
        flatInputData.insert(flatInputData.end(), data.begin(), data.end());
    }

    // Create a non-const copy of labels data
    std::vector<int64_t> nonConstLabels(labels.begin(), labels.end());

    // Define the shape for input
    const std::vector<int64_t> inputShape = {static_cast<int64_t>(inputData.size()), 6, 1};
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

    // Load checkpoint and initialize training session
    auto state = Ort::CheckpointState::LoadCheckpoint("./training_artifacts/checkpoint");
    auto training_session = Ort::TrainingSession(env, session_options, state, "./training_artifacts/training_model.onnx", "./training_artifacts/eval_model.onnx", "./training_artifacts/optimizer_model.onnx");

    // Generate training data
    auto trainingData = generateTrainingData();

    // Create batches
    int batchSize = 512;
    auto batches = createBatches(trainingData, batchSize);

    int numEpochs = 31;

    // Training loop
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        float totalEpochLoss = 0.0f;
        int numBatches = 0;

        for (auto& batch : batches) {
            auto& [batchInput, batchLabels] = batch;
            float batchLoss = trainStep(batchInput, batchLabels, training_session);
            totalEpochLoss += batchLoss;
            numBatches++;
        }

        // Average loss for the epoch
        float averageEpochLoss = totalEpochLoss / numBatches;

        // Print loss every 5 epochs
        if ((epoch) % 5 == 0) {
            std::cout << "Epoch " << (epoch + 1) << "/" << numEpochs << " - Loss: " << averageEpochLoss << std::endl;
        }

        // Additional code for model evaluation and checkpoint saving can be added here
    }

    // Export the trained model for inferencing
    std::string inference_model_path = "./inference_model.onnx";
    std::vector<std::string> output_node_names = {"my_output"}; // Replace with your actual output node names

    try {
        training_session.ExportModelForInferencing(inference_model_path, output_node_names);
        std::cout << "Model exported successfully to: " << inference_model_path << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "Error exporting model: " << e.what() << std::endl;
    }

    // Save the training checkpoint
    std::string path_to_checkpoint = "./training_checkpoint.chkpt";
    Ort::CheckpointState::SaveCheckpoint(state, path_to_checkpoint, false);

    return 0;
}

