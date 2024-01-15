//
//  Utils.swift
//  TestLSTM
//
//  Created by Baicen Xiao on 12/22/23.
//
import CoreML

func predict(m1: MLMultiArray) -> Int? {
    let model = try! lstm_model()
    do {
        let prediction = try model.prediction(x: m1).linear_0

        var topIndexAndValue: (index: Int, value: Float)?

        // Iterate through the MLMultiArray and store the values and their indices
        for i in 0..<prediction.count {
            if let value = prediction[i] as? Float {
                // Update the top value if this value is greater
                if topIndexAndValue == nil || value > topIndexAndValue!.value {
                    topIndexAndValue = (index: i, value: value)
                }
            }
        }

        // Return the top index
        return topIndexAndValue?.index

    } catch {
        print(error)
        return nil
    }
}
