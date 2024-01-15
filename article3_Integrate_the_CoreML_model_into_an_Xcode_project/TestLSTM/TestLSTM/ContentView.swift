//
//  ContentView.swift
//  TestLSTM
//
//  Created by Baicen Xiao on 12/22/23.
//
import SwiftUI
import CoreML


struct ContentView: View {
    @State private var arrayString: String = ""
    @State private var prediction: String = ""

    var body: some View {
        VStack {
            TextField("Enter an array like [1,2,3,4,5,6]", text: $arrayString)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()

            Button("Predict") {
                do {
                    let inputArray = try parseAndReshapeInput(from: arrayString)
                    let prediction = predict(m1: inputArray)
                    self.prediction = String(describing: prediction)
                } catch {
                    self.prediction = "Error: \(error.localizedDescription)"
                }
            }
            .padding()

            Text("Prediction: \(prediction)")
        }
        .padding()
    }

    private func parseAndReshapeInput(from string: String) throws -> MLMultiArray {
        // Parse the string to create an array of Float32
        let array = string
            .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
            .split(separator: ",")
            .compactMap { Float32($0.trimmingCharacters(in: .whitespaces)) }
        
        // Check if array size is correct
        guard array.count == 6 else {
            throw NSError(domain: "com.yourapp", code: 1, userInfo: [NSLocalizedDescriptionKey: "Array must have 6 elements"])
        }

        // Create a MLMultiArray
        let multiArray = try MLMultiArray(shape: [1, 6, 1], dataType: .float32)
        for (index, element) in array.enumerated() {
            multiArray[[0, index, 0] as [NSNumber]] = NSNumber(value: element)
        }

        return multiArray
    }
}

#Preview {
    ContentView()
}
