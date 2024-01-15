// Import required modules
import SwiftUI
import onnxruntime_objc
import Combine

struct ContentView: View {
    // State variables to store user input and prediction result
    @State private var arrayString: String = ""
    @State private var prediction: String = ""

    // Main body of the view
    var body: some View {
        VStack {
            // Text field for user to input array data
            TextField("Enter an array like [1,2,3,4,5,6]", text: $arrayString)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()

            // Button to trigger prediction
            Button("Predict") {
                predictWithONNXModel()
            }
            .padding()

            // Display the prediction result
            Text("Prediction: \(prediction)")
        }
        .padding()
    }

    // Function to handle prediction logic
    private func predictWithONNXModel() {
        do {
            // Parse and reshape the input array
            let inputArray = try parseAndReshapeInput(from: arrayString)
            // Perform prediction using the input array
            let topIndices = try predict(for: inputArray)
            // Update the prediction state variable
            self.prediction = String(describing: topIndices)
        } catch {
            // Handle and display any errors
            self.prediction = "Error: \(error.localizedDescription)"
        }
    }

    // Function to parse and reshape the input string into a 2D array
    private func parseAndReshapeInput(from string: String) throws -> [[Float]] {
        // Parse the string into an array of Float
        let array = string
            .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
            .split(separator: ",")
            .compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }

        // Ensure the array length is divisible by 6 (assuming each sample has 6 features)
        guard array.count % 6 == 0 else {
            throw NSError(domain: "com.yourapp", code: 1, userInfo: [NSLocalizedDescriptionKey: "Array size must be a multiple of 6"])
        }

        // Reshape the flat array into a 2D array with each sub-array containing 6 elements
        let reshapedArray = stride(from: 0, to: array.count, by: 6).map {
            Array(array[$0..<min($0 + 6, array.count)])
        }

        return reshapedArray
    }
}

// SwiftUI preview for the ContentView
#Preview {
    ContentView()
}
