# GEMINI.md - Project Overview

## Project Overview

This project is a C++ application named `rk-app`, designed for object detection and PID control, likely targeting the RK3588 platform. It's a well-structured project that uses CMake for building and supports cross-compilation for arm64. The project also includes Python scripts for tuning and plotting, indicating a mix of C++ for performance-critical tasks and Python for higher-level control and analysis.

The project is divided into two main parts:

1.  **PID Controller:** A library for PID control, with both C++ and C implementations. It includes a benchmark application to evaluate the performance of the PID controller.
2.  **Object Detection Pipeline:** A complete object detection pipeline that can use different inference engines (ONNX and RKNN) and input sources (folder, video, GigE camera). The pipeline is configurable through YAML files and can be controlled via a command-line interface.

## Building and Running

The project uses CMake with presets for different configurations. The main build and run commands can be found in the `.vscode/tasks.json` and `CMakePresets.json` files.

### Building

*   **x86 (Debug):**
    ```bash
    cmake --preset x86-debug
    cmake --build build/x86-debug
    ```
*   **arm64 (Release):**
    ```bash
    cmake --preset arm64
    cmake --build build/arm64
    ```

### Running

*   **rk_app (main application):**
    ```bash
    ./build/x86-debug/rk_app
    ```
*   **detect_cli (object detection):**
    ```bash
    ./build/x86-debug/detect_cli --cfg config/detect.yaml --source <path_to_images>
    ```
*   **pid_bench (PID benchmark):**
    ```bash
    ./build/x86-debug/pid_bench
    ```

### Testing

The project uses CTest for testing.

```bash
ctest --preset x86-debug
```

## Development Conventions

*   **Code Style:** The project uses `.clang-format` and `.clang-tidy` for code formatting and linting, which suggests a consistent code style is enforced.
*   **Testing:** The project has a `tests` directory and uses GoogleTest for unit testing. This indicates a commitment to testing and code quality.
*   **Configuration:** The application is configured using YAML files, which allows for easy modification of parameters without recompiling the code.
*   **Modularity:** The code is well-structured and modular, with different components (capture, inference, postprocessing, output) separated into their own libraries. This makes the code easier to understand, maintain, and extend.
