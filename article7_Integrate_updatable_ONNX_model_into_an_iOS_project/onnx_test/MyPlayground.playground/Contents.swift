import UIKit
import onnxruntime_training_objc

let ortEnv: ORTEnv
let ortSession: ORTSession

enum TrainerError: Error {
        case Error(_ message: String)
    }

ortEnv = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
guard let modelPath = Bundle.main.path(forResource: "inference_trained", ofType: "onnx") else {
            throw TrainerError.Error("Failed to find training model file.")
        }


ortSession = try ORTSession(env: ortEnv, modelPath: modelPath, sessionOptions: nil)


let array: [[Float]] = [
    [0.5764202, -1.9590191, 0.9312385],
    [0.26379728, 0.9968908, 0.67580676],
    [-0.42796516, -0.5641794, 1.5809623]
]

// Flatten the 2D array into a 1D array
let flattenedArray = array.flatMap { $0 }

// Convert the array of floats into Data
//let inputData = Data(buffer: UnsafeBufferPointer(start: flattenedArray, count: flattenedArray.count))
let inputData = flattenedArray.withUnsafeBufferPointer { Data(buffer: $0) }


let numSamples = array.count
let numFeatures = array.first?.count ?? 0

// Create the input shape
let inputShape: [NSNumber] = [NSNumber(value: numSamples), NSNumber(value: numFeatures)]

print(inputData)
print(inputShape)

do {
    // Create an ORTValue object for the input
    let input = try ORTValue(
        tensorData: NSMutableData(data: inputData),
        elementType: ORTTensorElementDataType.float,
        shape: inputShape)

    // Run the model
    let outputs = try ortSession.run(
        withInputs: ["input": input],
        outputNames: ["output"],
        runOptions: nil)

    // Extract the output
    guard let output = outputs["output"] else {
        throw NSError(domain: "VoiceIdentifierError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Failed to get model output from inference."])
    }
    
    let outputData = try output.tensorData() as Data
    print(outputData)
    let floatArray = outputData.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) -> [Float] in
        let bufferPointer = pointer.bindMemory(to: Float.self)
        return Array(bufferPointer)
    }
    print(floatArray)
    // Process the output as needed
    // This part depends on the expected format and structure of your model's output

} catch {
    print("An error occurred during model inference: \(error)")
}

