//
//  ContentView.swift
//  onnx_test
//
//  Created by Baicen Xiao on 12/26/23.
//

import SwiftUI
import Combine

struct ContentView: View {
    @ObservedObject var trainer = TrainerViewModel()

    var body: some View {
        VStack {
            // Graph to display training loss
            GraphView(lossData: trainer.lossData)
                .frame(height: 300)
                .padding()

            // Start Training Button
            Button("Start Training") {
                trainer.startTraining()
            }
            .padding()
        }
    }
}


struct GraphView: View {
    var lossData: [Float]
    let yAxisSteps: Int = 5
    let maxYValue: Float = 0.1
    let xAxisSteps: Int = 20 // Set to 5 fixed labels on X-axis

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Draw Axes
                drawAxes(in: geometry.size)

                // Draw Y-Axis Labels
                ForEach(0...yAxisSteps, id: \.self) { step in
                    let stepValue = maxYValue / Float(yAxisSteps) * Float(step)
                    let yPosition = (1 - CGFloat(step) / CGFloat(yAxisSteps)) * geometry.size.height
                    Text(String(format: "%.2f", stepValue))
                        .position(x: geometry.size.width * 0.05, y: yPosition)
                        .font(.caption)
                }

                // Draw X-Axis Labels (Fixed)
                ForEach(0..<xAxisSteps, id: \.self) { index in
                    let xPosition = geometry.size.width * 0.1 + (geometry.size.width * 0.9 / CGFloat(xAxisSteps) * CGFloat(index))
                    Text("\(index + 1)")
                        .position(x: xPosition, y: geometry.size.height + 10) // Position below the X axis
                        .font(.caption)
                }

                // Draw the Loss Line
                Path { path in
                    for (index, loss) in lossData.enumerated() {
                        let xPosition = geometry.size.width * 0.1 + (geometry.size.width * 0.9 / CGFloat(lossData.count) * CGFloat(index))
                        let yPosition = (1 - CGFloat(loss / maxYValue)) * geometry.size.height
                        if index == 0 {
                            path.move(to: CGPoint(x: xPosition, y: yPosition))
                        } else {
                            path.addLine(to: CGPoint(x: xPosition, y: yPosition))
                        }
                    }
                }
                .stroke(Color.blue, lineWidth: 2)
            }
        }
    }

    private func drawAxes(in size: CGSize) -> some View {
        Path { path in
            // Draw Y axis
            path.move(to: CGPoint(x: size.width * 0.1, y: 0))
            path.addLine(to: CGPoint(x: size.width * 0.1, y: size.height))

            // Draw X axis
            path.move(to: CGPoint(x: size.width * 0.1, y: size.height))
            path.addLine(to: CGPoint(x: size.width, y: size.height))
        }
        .stroke(Color.black, lineWidth: 1)
    }
}



class TrainerViewModel: ObservableObject {
    @Published var lossData: [Float] = []

    func startTraining() {
        self.lossData.removeAll()
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try trainModel { epoch, batchIndex, loss in
                    DispatchQueue.main.async {
                        self.lossData.append(loss)
                        print("Epoch: \(epoch), Batch: \(batchIndex), Loss: \(loss)")
                    }
                }
            } catch {
                print("Training failed with error: \(error)")
            }
        }
    }
}


#Preview {
    ContentView()
}
