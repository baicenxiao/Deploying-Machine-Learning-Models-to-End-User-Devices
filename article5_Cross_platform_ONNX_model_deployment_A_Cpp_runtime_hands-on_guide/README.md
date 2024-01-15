
## How to use

1. Here is an example of using ONNX C++ runtime to run inference using the same input as above:
   - Build docker image and run docker container:
       ```bash
       docker build -t onnxruntime_image_test .
       docker run -it --rm --name onnxruntime_container_test -v /path/on/host/to/onnx/model:/model onnxruntime_image_test
       ```
       If you are using MacOS with Apple Silicon chip, run the following build instead
      ```
      docker buildx build --platform linux/amd64 -t onnxruntime_image_test .
      ```

   - In docker container, compile one of the `test_onnx_cxx.cpp`
       ```
       g++ test_onnx_cxx.cpp -o test_onnx_cxx -I/onnxruntime/onnxruntime-linux-x64-1.16.1/include -L/onnxruntime/onnxruntime-linux-x64-1.16.1/lib -lonnxruntime
       ```

   - Run the compiled file
       ```
       ./test_onnx_cxx
       ```
     The output looks like ![Alt text](image.png)
