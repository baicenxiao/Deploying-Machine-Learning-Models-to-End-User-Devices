//
//  utils.swift
//  TestLSTM
//
//  Created by Baicen Xiao on 12/23/23.
//

import onnxruntime_objc

enum TrainerError: Error {
    case modelFileNotFound
    case modelInferenceFailed(String)
}


func predict(for inputArray: [[Float]]) throws -> [Int] {
    
    guard let modelPath = Bundle.main.path(forResource: "lstm_model", ofType: "onnx") else {
        throw TrainerError.modelFileNotFound
    }
    
    let ortEnv = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
    let ortSession = try ORTSession(env: ortEnv, modelPath: modelPath, sessionOptions: nil)

    let flattenedArray = inputArray.flatMap { $0 }
    let inputData = flattenedArray.withUnsafeBufferPointer { Data(buffer: $0) }

    let numSamples = inputArray.count
    let numFeatures = inputArray.first?.count ?? 0
    let inputShape: [NSNumber] = [NSNumber(value: numSamples), NSNumber(value: numFeatures), NSNumber(value: 1)]

    let input = try ORTValue(tensorData: NSMutableData(data: inputData), elementType: ORTTensorElementDataType.float, shape: inputShape)

    let outputs = try ortSession.run(withInputs: ["seq_input": input], outputNames: ["my_output"], runOptions: nil)

    guard let output = outputs["my_output"], let outputData = try? output.tensorData() as Data else {
        throw TrainerError.modelInferenceFailed("Failed to get model output from inference.")
    }

    let predictions = outputData.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) -> [Float] in
        let bufferPointer = pointer.bindMemory(to: Float.self)
        return Array(bufferPointer)
    }

    // Assuming the output is a 2D array (batch_size x num_outputs)
    let reshapedPredictions = stride(from: 0, to: predictions.count, by: 3).map {
        Array(predictions[$0..<$0 + 3])
    }

    return reshapedPredictions.map { $0.firstIndex(of: $0.max()!) ?? -1 }
}
