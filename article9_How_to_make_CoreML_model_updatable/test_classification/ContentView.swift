import SwiftUI
import CoreML

struct ContentView: View {
    @State private var isTraining = false
    @State private var isTesting = false
    @State private var trainingLossValues: [Double] = []

    var body: some View {
        VStack {
            // Display training status
            if isTraining {
                Text("Training in progress...")
            } else {
                Text("Ready to train model")
            }

            // Button to start training
            Button(action: {
                self.startTraining()
            }) {
                Text("Train  Model")
                    .padding()
                    .foregroundColor(.white)
                    .background(isTraining ? Color.gray : Color.blue)
                    .cornerRadius(40)
            }
            .disabled(isTraining) // Disable the button when training or testing is in progress

            // Display testing status
            if isTesting {
                Text("Testing in progress...")
            } else {
                Text("Ready to test model")
            }

            // Button to start testing
            Button(action: {
                self.startTesting()
            }) {
                Text("Test Model")
                    .padding()
                    .foregroundColor(.white)
                    .background(isTesting ? Color.gray : Color.green)
                    .cornerRadius(40)
            }
            .disabled(isTesting) // Disable the button when testing or training is in progress
            
            LossCurveView(lossValues: trainingLossValues)
                            .frame(height: 200) // Set the desired height for the loss curve
                            .padding()
                            .background(Color.black.opacity(0.1))
                            .cornerRadius(10)
        }
        .padding()
    }

    private func updateTrainingLoss(newLossValue: Double) {
            DispatchQueue.main.async {
                self.trainingLossValues.append(newLossValue)
            }
        }

    private func startTraining() {
        self.isTraining = true
        trainingLossValues.removeAll()
        DispatchQueue.global(qos: .userInitiated).async {
            trainModel(updateLoss: { lossValue in
                DispatchQueue.main.async {
                    self.updateTrainingLoss(newLossValue: lossValue)
                }
            }, completion: {
                DispatchQueue.main.async {
                    self.isTraining = false
                }
            })
        }
    }

    private func startTesting() {
        isTesting = true
        DispatchQueue.global(qos: .userInitiated).async {
            testModel()
            DispatchQueue.main.async {
                self.isTesting = false
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
