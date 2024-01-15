import CoreML

import GameKit

func generateSampleData(numSamples: Int, seed: UInt64) -> ([MLMultiArray], [MLMultiArray]) {
    var inputArray = [MLMultiArray]()
    var outputArray = [MLMultiArray]()

    let randomSource = GKLinearCongruentialRandomSource(seed: seed)
    let randomDistribution = GKRandomDistribution(randomSource: randomSource, lowestValue: 0, highestValue: 10)

    for _ in 0..<numSamples {
        do {
            let input = try MLMultiArray(shape: [1, 3], dataType: .float32)
            let output = try MLMultiArray(shape: [1], dataType: .int32)

            var sumInput: Float = 0
            for i in 0..<input.count {
                let inputValue = Float(randomDistribution.nextInt())
                input[i] = NSNumber(value: inputValue)
                sumInput += inputValue
            }

            // Assign a class based on the sum of inputs
            let outputClass = Int32(sumInput/6.0) % 5
            output[0] = NSNumber(value: outputClass)

            inputArray.append(input)
            outputArray.append(output)
        } catch {
            print("Error occurred while creating MLMultiArrays: \(error)")
        }
    }
    
    return (inputArray, outputArray)
}


func argmax(multiArray: MLMultiArray) -> Int {
    let length = multiArray.count
    let ptr = UnsafeMutablePointer<Float>(OpaquePointer(multiArray.dataPointer))
    var maxValue: Float = ptr[0]
    var maxIndex: Int = 0
    
    for i in 1..<length {
        if ptr[i] > maxValue {
            maxValue = ptr[i]
            maxIndex = i
        }
    }
    
    return maxIndex
}


func computeMetrics(model: MLModel, data: ([MLMultiArray], [MLMultiArray])) -> (loss: Double, accuracy: Double) {
    let (inputData, outputData) = data
    var totalLoss: Double = 0
    var correctPredictions: Int = 0

    for (index, input) in inputData.enumerated() {
        let output = outputData[index]

        if let prediction = try? model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: input)])),
           let predictedOutputProb = prediction.featureValue(for: "output_prob")?.multiArrayValue {

            let trueClass = output[0].intValue
            let predictedClass = argmax(multiArray: predictedOutputProb)
            correctPredictions += (trueClass == predictedClass) ? 1 : 0

            // Calculate cross-entropy loss
            let predictedProb = predictedOutputProb[trueClass]
            totalLoss += -log(max(Double(predictedProb.doubleValue), 1e-10))
        }
    }
    let accuracy = Double(correctPredictions) / Double(inputData.count)
    return (totalLoss / Double(inputData.count), accuracy)
}




func trainModel(updateLoss: @escaping (Double) -> Void, completion: @escaping () -> Void) {
    // Load the updatable model
    guard let updatableModelURL = Bundle.main.url(forResource: "updatable_classification", withExtension: "mlmodelc") else {
        print("Failed to load the updatable model")
        return
    }

    // Generate sample data
    let (inputData, outputData) = generateSampleData(numSamples: 500, seed: 8)
    let validationData = generateSampleData(numSamples: 100, seed: 18)

    // Create an MLArrayBatchProvider from the sample data
    var featureProviders = [MLFeatureProvider]()
    for (index, input) in inputData.enumerated() {
        let output = outputData[index]
        let dataPointFeatures: [String: MLFeatureValue] = [
            "x": MLFeatureValue(multiArray: input), // Make sure "input" matches the input feature name in your model
            "output_prob_true": MLFeatureValue(multiArray: output) // Make sure "output_true" matches the expected output feature name in your model for training
        ]
        if let provider = try? MLDictionaryFeatureProvider(dictionary: dataPointFeatures) {
            featureProviders.append(provider)
        }
    }
    let batchProvider = MLArrayBatchProvider(array: featureProviders)

    // Define progress handlers
    var lossValues: [Double] = []
    var accuracyValues: [Double] = []
    var validationMetrics: [(loss: Double, accuracy: Double)] = []

    let progressHandlers = MLUpdateProgressHandlers(forEvents: [.trainingBegin, .epochEnd],
        progressHandler: { context in
            switch context.event {
                case .trainingBegin:
                    print("Training began.")
                case .epochEnd:
                    let loss = context.metrics[.lossValue] as! Double
                    lossValues.append(loss)
                    let (validationLoss, validationAccuracy) = computeMetrics(model: context.model, data: validationData)
                    validationMetrics.append((validationLoss, validationAccuracy))
                    let (computedTrainLoss, computedTrainAccuracy) = computeMetrics(model: context.model, data: (inputData, outputData))
                    accuracyValues.append(computedTrainAccuracy)
                    updateLoss(computedTrainLoss)
                    print("Epoch \(context.metrics[.epochIndex]!) ended. Training Loss: \(loss), Computed Training loss: \(computedTrainLoss), Training Accuracy: \(computedTrainAccuracy), Validation Loss: \(validationLoss), Validation Accuracy: \(validationAccuracy)")
                default:
                    break
            }
        },
        completionHandler: { context in
            if let error = context.task.error {
                print("Update task failed with error: \(error)")
            } else {
                let updatedModel = context.model
                do {
                    let fileManager = FileManager.default
                    let documentDirectory = try fileManager.url(for: .documentDirectory, in: .userDomainMask, appropriateFor:nil, create:true)
                    let fileURL = documentDirectory.appendingPathComponent("updated_classification.mlmodelc")
                    try updatedModel.write(to: fileURL)
                    print("Model updated and saved successfully to \(fileURL)")
                } catch {
                    print("Failed to save the updated model: \(error)")
                }
            }
        completion()
        }
    )

    // Create an update task with progress handlers
    let updateTask = try! MLUpdateTask(forModelAt: updatableModelURL,
                                       trainingData: batchProvider,
                                       configuration: nil,
                                       progressHandlers: progressHandlers)

    // Start the update task
    updateTask.resume()
}



func testModel() {
    // Generate test data
    let (testInputs, testOutputs) = generateSampleData(numSamples: 10, seed: 188)

    // Load the updated model
    let fileManager = FileManager.default
    guard let documentDirectory = try? fileManager.url(for: .documentDirectory, in: .userDomainMask, appropriateFor:nil, create:false),
          let model = try? MLModel(contentsOf: documentDirectory.appendingPathComponent("updated_classification.mlmodelc")) else {
        print("Failed to load the updated model")
        return
    }

    // Make predictions
    for i in 0..<testInputs.count {
        let input = testInputs[i]
        let expectedOutputClass = testOutputs[i][0].intValue // Expected class

        do {
            let inputFeatureProvider = try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: input)])
            let prediction = try model.prediction(from: inputFeatureProvider)

            if let predictedOutputProbArray = prediction.featureValue(for: "output_prob")?.multiArrayValue {
                let predictedClass = argmax(multiArray: predictedOutputProbArray) // Use the custom argmax function
                print("Sample \(i+1): \(input)")
                print("Predicted Class: \(predictedClass)")
                print("Expected Class: \(expectedOutputClass)")
            } else {
                print("Failed to extract prediction for sample \(i+1)")
            }
        } catch {
            print("Failed to make prediction for sample \(i+1): \(error.localizedDescription)")
        }
    }
}
