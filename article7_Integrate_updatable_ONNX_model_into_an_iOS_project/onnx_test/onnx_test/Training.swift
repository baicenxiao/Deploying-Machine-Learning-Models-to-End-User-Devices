//
//  Training.swift
//  onnx_test
//
//  Created by Baicen Xiao on 12/26/23.
//

import Foundation
import onnxruntime_training_objc

// Define an enum for handling training errors
enum TrainerError: Error {
    case Error(_ message: String)
}

// Helper function to convert a 2D array of features (Float) to an ORTValue (used in ONNX Runtime)
private func getORTValue(forFeatures features: [[Float]]) throws -> ORTValue {
    // Flatten the 2D array into a 1D array
    let flattenedArray = features.flatMap { $0 }
    // Convert the array to Data
    let tensorData = Data(buffer: UnsafeBufferPointer(start: flattenedArray, count: flattenedArray.count))
    // Define the shape of the tensor
    let inputShape: [NSNumber] = [features.count as NSNumber, features[0].count as NSNumber, 1 as NSNumber]

    // Create and return the ORTValue
    return try ORTValue(
        tensorData: NSMutableData(data: tensorData),
        elementType: ORTTensorElementDataType.float,
        shape: inputShape
    )
}

// Helper function to convert a 1D array of labels (Int) to an ORTValue
private func getORTValue(forLabels1D labels: [Int]) throws -> ORTValue {
    // Convert the array of Int to Int64
    let int64Labels = labels.map { Int64($0) }
    // Convert the array to Data
    let tensorData = Data(buffer: UnsafeBufferPointer(start: int64Labels, count: int64Labels.count))
    // Define the shape for a 1D tensor
    let inputShape: [NSNumber] = [labels.count as NSNumber]

    // Create and return the ORTValue
    return try ORTValue(
        tensorData: NSMutableData(data: tensorData),
        elementType: ORTTensorElementDataType.int64,
        shape: inputShape
    )
}

// Main function for training the model
func trainModel(progressReporter: (_ epoch: Int, _ batchIndex: Int, _ loss: Float) -> Void) throws {
    // Initialize ONNX Runtime environment
    let ortEnv = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
    
    // Load paths for training, evaluation, optimizer models and checkpoint
    // (typically, these paths should be passed as parameters or configured externally)
    // Throw errors if any paths are not found
    guard let trainingModelPath = Bundle.main.path(forResource: "training_model", ofType: "onnx"),
          let evalModelPath = Bundle.main.path(forResource: "eval_model",ofType: "onnx"),
          let optimizerPath = Bundle.main.path(forResource: "optimizer_model", ofType: "onnx"),
          let checkpointPath = Bundle.main.path(forResource: "checkpoint", ofType: nil) else {
        throw TrainerError.Error("Failed to find required model file or checkpoint.")
    }

    // Load the checkpoint
    let checkpoint = try ORTCheckpoint(path: checkpointPath)

    // Create a training session with the loaded paths
    let trainingSession = try ORTTrainingSession(env: ortEnv, sessionOptions: ORTSessionOptions(), checkpoint: checkpoint, trainModelPath: trainingModelPath, evalModelPath: evalModelPath, optimizerModelPath: optimizerPath)

    // Define the number of training epochs
    let kNumEpochs: Int = 20

    // Generate or load your training data
    let (xTrain, yTrain) = generateTrainingData()

    // Split training data into batches
    let batchSize = 512 // Batch size
    let batches = createBatches(fromFeatures: xTrain, labels: yTrain, batchSize: batchSize)

    // Training loop: iterate through epochs and batches
    for epoch in 0..<kNumEpochs {
        for (i, batch) in batches.enumerated() {
            let (batchInput, batchLabels) = batch
            // Perform a training step and obtain the loss
            let loss = try trainStep(trainingSession: trainingSession, inputData: batchInput, labels: batchLabels)

            // Call the progress reporter with current epoch, batch index, and loss
            progressReporter(epoch, i, loss)
        }
    }
}

// Function to perform a single training step
func trainStep(trainingSession: ORTTrainingSession, inputData: [[Float]], labels: [Int]) throws -> Float {
    // Convert input data and labels to ORTValue
    let xORTValue = try getORTValue(forFeatures: inputData)
    let yORTValue = try getORTValue(forLabels1D: labels)

    // Perform the training step and get outputs
    let inputs = [xORTValue, yORTValue]
    let outputs = try trainingSession.trainStep(withInputValues: inputs)

    // Extract and return the loss from the outputs
    guard let lossValue = outputs.first else {
        throw TrainerError.Error("Failed to get loss value from training step.")
    }

    let lossData = try lossValue.tensorData() as Data
    let lossArray = lossData.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) -> [Float] in
        let bufferPointer = pointer.bindMemory(to: Float.self)
        return Array(bufferPointer)
    }

    // Update model parameters and reset gradients
    try trainingSession.optimizerStep()
    try trainingSession.lazyResetGrad()
    
    // Return the first element of lossArray as the loss, or 0 if empty
    return lossArray.first ?? 0
}


