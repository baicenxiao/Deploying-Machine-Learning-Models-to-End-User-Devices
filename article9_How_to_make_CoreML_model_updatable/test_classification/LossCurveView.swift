//
//  LossCurveView.swift
//  test_classification
//
//  Created by Baicen Xiao on 11/27/23.
//

import SwiftUI

struct LossCurveView: View {
    var lossValues: [Double]

    var body: some View {
        GeometryReader { geometry in
            Path { path in
                for (index, loss) in lossValues.enumerated() {
                    let xPosition = geometry.size.width * Double(index) / Double(lossValues.count)
                    let yPosition = (1 - loss) * geometry.size.height // Assuming loss is normalized

                    if index == 0 {
                        path.move(to: CGPoint(x: xPosition, y: yPosition))
                    } else {
                        path.addLine(to: CGPoint(x: xPosition, y: yPosition))
                    }
                }
            }
            .stroke(Color.red, lineWidth: 2)
        }
    }
}

//#Preview { 
//    LossCurveView()
//}
