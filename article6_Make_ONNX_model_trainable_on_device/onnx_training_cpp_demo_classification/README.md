## This repo shows how to convert model from pytorch to onnx and perform on-device training with onnxruntime c++ api.

### Prerequisites
`torch==2.1.0` and `onnxruntime==1.16.1` are used in this repo.

### Usage
1. Use `Prepare_trainable_onnx_model.ipynb` to convert pytorch model to trainable onnx model.
2. Build docker image
   - Build docker container and run docker container:
       ```bash
       docker build -t onnxruntime_training_image .
       docker run -it --rm --name onnxruntime_container -v /path/on/host/to/onnx/model:/model onnxruntime_training_image
       ```
       If you are using MacOS with Apple Silicon chip, run the following build instead
      ```
      docker buildx build --platform linux/amd64 -t onnxruntime_training_image .
      ```

   - In docker container, compile the `train_onnx.cpp`
       ```
       g++ train_onnx.cpp -o train_onnx -I/onnxruntime/onnxruntime-training-linux-x64-1.16.1/include -I/vcpkg/installed/x64-linux/include -L/onnxruntime/onnxruntime-training-linux-x64-1.16.1/lib -lonnxruntime -std=c++17
       ```
3. Run `./train_onnx` to perform on-device training and save trained model.