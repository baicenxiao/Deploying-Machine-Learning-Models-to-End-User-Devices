//
//  SaveModel.swift
//  onnx_test
//
//  Created by Baicen Xiao on 12/26/23.
//

import Foundation
import onnxruntime_training_objc

func saveTrainedModel(trainingSession: ORTTrainingSession) throws {
    guard let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
        throw TrainerError.Error("Documents directory not found.")
    }

    let modelPath = documentsDirectory.appendingPathComponent("inference_trained.onnx").path
    try trainingSession.exportModelForInference(withOutputPath: modelPath, graphOutputNames: ["my_output"])
}

// Call this function with a valid ORTTrainingSession instance after training is complete.
