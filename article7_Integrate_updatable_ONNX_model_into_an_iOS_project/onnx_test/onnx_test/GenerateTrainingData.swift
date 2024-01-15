//
//  GenerateTrainingData.swift
//  onnx_test
//
//  Created by Baicen Xiao on 12/26/23.
//

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
