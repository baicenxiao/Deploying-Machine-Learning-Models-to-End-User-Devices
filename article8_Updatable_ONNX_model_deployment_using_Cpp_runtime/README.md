### Usage
1. Build docker image
   - Build docker container and run docker container:
       ```bash
       docker build -t onnxruntime_training_image_test .
       docker run -it --rm --name onnxruntime_container_training_test -v /path/on/host/to/onnx/model:/model onnxruntime_training_image_test
       ```
       If you are using MacOS with Apple Silicon chip, run the following build instead
      ```
      docker buildx build --platform linux/amd64 -t onnxruntime_training_image_test .
      ```

   - In docker container, compile the `train_onnx.cpp`
       ```
       g++ train_onnx.cpp -o train_onnx -I/onnxruntime/onnxruntime-training-linux-x64-1.16.1/include -L/onnxruntime/onnxruntime-training-linux-x64-1.16.1/lib -lonnxruntime -std=c++17
       ```
2. Run `./train_onnx` to perform on-device training and save trained model.