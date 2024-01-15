import Foundation
import onnxruntime_training_objc

let ortEnv: ORTEnv
let trainingSession: ORTTrainingSession
let checkpoint: ORTCheckpoint
let kNumEpochs: Int = 20

let kUserIndex: Int64 = 1
let kOtherIndex: Int64 = 0

enum TrainerError: Error {
    case Error(_ message: String)
}

ortEnv = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
        
// get path for artifacts
guard let trainingModelPath = Bundle.main.path(forResource: "training_model", ofType: "onnx") else {
    throw TrainerError.Error("Failed to find training model file.")
}

guard let evalModelPath = Bundle.main.path(forResource: "eval_model",ofType: "onnx") else {
    throw TrainerError.Error("Failed to find eval model file.")
}

guard let optimizerPath = Bundle.main.path(forResource: "optimizer_model", ofType: "onnx") else {
    throw TrainerError.Error("Failed to find optimizer model file.")
}

guard let checkpointPath = Bundle.main.path(forResource: "checkpoint", ofType: nil) else {
    throw TrainerError.Error("Failed to find checkpoint file.")
}
        
checkpoint = try ORTCheckpoint(path: checkpointPath)

trainingSession = try ORTTrainingSession(env: ortEnv, sessionOptions: ORTSessionOptions(), checkpoint: checkpoint, trainModelPath: trainingModelPath, evalModelPath: evalModelPath, optimizerModelPath: optimizerPath)



import Foundation

func generateTrainingData(dataSize: Int = 1000) -> (x: [[Float]], y: [Int]) {
    var x = [[Float]]()
    var y = [Int]()

    for _ in 0..<dataSize {
        let randomValues = (0..<6).map { _ in Float.random(in: 0.0...10.0) }
        x.append(randomValues)

        let sum = randomValues.reduce(0, +)
        let yValue = Int(sum / 20)
        y.append(yValue)
    }

    return (x, y)
}


// Generate training data
//let (xTrain, yTrain) = generateTrainingData()
private func getORTValue(forFeatures features: [[Float]]) throws -> ORTValue {
    let flattenedArray = features.flatMap { $0 }
    let tensorData = Data(buffer: UnsafeBufferPointer(start: flattenedArray, count: flattenedArray.count))
    let inputShape: [NSNumber] = [features.count as NSNumber, features[0].count as NSNumber, 1 as NSNumber]

    return try ORTValue(
        tensorData: NSMutableData(data: tensorData),
        elementType: ORTTensorElementDataType.float,
        shape: inputShape
    )
}


//private func getORTValue(forLabels2D labels: [[Int]]) throws -> ORTValue {
//    let flattenedArray = labels.flatMap { $0 }
//    let tensorData = Data(buffer: UnsafeBufferPointer(start: flattenedArray, count: flattenedArray.count))
//    let inputShape: [NSNumber] = [labels.count as NSNumber, 1]
//
//    return try ORTValue(
//        tensorData: NSMutableData(data: tensorData),
//        elementType: ORTTensorElementDataType.float,
//        shape: inputShape
//    )
//}

//private func getORTValue(forLabels2D labels: [[Int]]) throws -> ORTValue {
//    let flattenedArray = labels.flatMap { $0.map { Int64($0) } } // Convert to Int64
//    let tensorData = Data(buffer: UnsafeBufferPointer(start: flattenedArray, count: flattenedArray.count))
//    let inputShape: [NSNumber] = [labels.count as NSNumber, 1]
//
//    return try ORTValue(
//        tensorData: NSMutableData(data: tensorData),
//        elementType: ORTTensorElementDataType.int64, // Use Int64 data type
//        shape: inputShape
//    )
//}
private func getORTValue(forLabels1D labels: [Int]) throws -> ORTValue {
    let int64Labels = labels.map { Int64($0) } // Convert to Int64
    let tensorData = Data(buffer: UnsafeBufferPointer(start: int64Labels, count: int64Labels.count))
    let inputShape: [NSNumber] = [labels.count as NSNumber] // Shape for 1D tensor

    return try ORTValue(
        tensorData: NSMutableData(data: tensorData),
        elementType: ORTTensorElementDataType.int64,
        shape: inputShape
    )
}



// Generate training data
let (xTrain, yTrain) = generateTrainingData()

// Function to create batches from training data
func createBatches(fromFeatures features: [[Float]], labels: [Int], batchSize: Int) -> [([[Float]], [Int])] {
    var batches = [([[Float]], [Int])]()
    
    for start in stride(from: 0, to: features.count, by: batchSize) {
        let end = min(start + batchSize, features.count)
        let batchFeatures = Array(features[start..<end])
        let batchLabels = Array(labels[start..<end])
        batches.append((batchFeatures, batchLabels))
    }

    return batches
}

// Create batches
let batchSize = 32 // Define your batch size here
let batches = createBatches(fromFeatures: xTrain, labels: yTrain, batchSize: batchSize)

//func adjustLabelsTo2D(_ labels: [Int]) -> [[Int]] {
//    return labels.map { [$0] }
//}


func trainStep(inputData: [[Float]], labels: [Int]) throws -> Float {
    let xORTValue = try getORTValue(forFeatures: inputData)
//    let adjustedLabels = adjustLabelsTo2D(labels)
//    let yORTValue = try getORTValue(forLabels2D: adjustedLabels)
    let yORTValue = try getORTValue(forLabels1D: labels)

    let inputs = [xORTValue, yORTValue]
    let outputs = try trainingSession.trainStep(withInputValues: inputs)
    // Assuming the first output is the loss
    guard let lossValue = outputs.first else {
        throw NSError(domain: "TrainingError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to get loss value from training step."])
    }

    let lossData = try lossValue.tensorData() as Data
    let lossArray = lossData.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) -> [Float] in
        let bufferPointer = pointer.bindMemory(to: Float.self)
        return Array(bufferPointer)
    }

    
    // Update the model parameters
    try trainingSession.optimizerStep()

    // Reset the gradients
    try trainingSession.lazyResetGrad()
    
    // Assuming the loss is a single float value
    return lossArray.first ?? 0
}



func train() throws {
    for epoch in 0..<kNumEpochs {
        print("Epoch: \(epoch)")

        for (i, batch) in batches.enumerated() {
            let (batchInput, batchLabels) = batch
            let loss = try trainStep(inputData: batchInput, labels: batchLabels)
            print("Finished training on batch \(i) with loss: \(loss)")
        }
    }
}


try train()

let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first

//// Check if the documents directory is available
if let documentsDirectory = documentsDirectory {
    // You can use documentsDirectory here
    // For example, appending a file path
    let modelPath = documentsDirectory.appendingPathComponent("inference_trained.onnx").path
    // Use the filePath as needed
    try trainingSession.exportModelForInference(withOutputPath: modelPath, graphOutputNames: ["my_output"])

} else {
    // Handle the error if the documents directory was not found
    print("Documents directory not found.")
}
//
