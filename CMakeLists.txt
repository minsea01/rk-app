cmake_minimum_required(VERSION 3.16)
project(rk_app LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 导出 compile_commands.json 供 VS Code 智能感知
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 可选构建选项：Sanitizers 与 覆盖率（仅 x86 Debug 推荐）
option(ENABLE_SANITIZERS "Enable ASan/UBSan on x86 Debug" ON)
option(ENABLE_COVERAGE "Enable coverage flags on x86 Debug" OFF)

# Object detection pipeline options
option(ENABLE_ONNX "Enable ONNXRuntime engine" ON)
option(ENABLE_RKNN "Enable RKNN engine" OFF)
option(ENABLE_RGA "Enable RGA acceleration (if available)" ON)
set(RKNN_HOME "/opt/rknpu2" CACHE PATH "RKNN SDK root")

if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i.86)")
  if (ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
  endif()
  if (ENABLE_COVERAGE)
    add_compile_options(--coverage)
    add_link_options(--coverage)
  endif()
endif()

# RPATH 设置：部署/安装后可在相对路径下找到依赖的 .so
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# 主程序
add_executable(rk_app src/main.cpp)
if(YAML_CPP_FOUND)
  target_link_libraries(rk_app PRIVATE ${YAML_CPP_LIBRARIES})
  target_include_directories(rk_app PRIVATE ${YAML_CPP_INCLUDE_DIRS})
  target_compile_definitions(rk_app PRIVATE RKAPP_HAVE_YAMLCPP=1)
else()
  message(WARNING "yaml-cpp not found; rk_app will use minimal fallback config parser")
endif()

# 示例算法库: PID
add_library(pid_lib src/pid.cpp include/pid.h)
target_include_directories(pid_lib PUBLIC include)

# 链接库到主程序
target_link_libraries(rk_app PRIVATE pid_lib)

# C 版 PID 控制器库（来自嵌入式实现，提供 micros() 适配）
add_library(pid_controller_c STATIC
  src/pid_controller.c
  src/drivers/time_service.c
)
target_include_directories(pid_controller_c PUBLIC include)
if(UNIX)
  target_link_libraries(pid_controller_c PUBLIC m)
endif()
# 链接到主程序（即使未直接调用也保证编译通过）
target_link_libraries(rk_app PRIVATE pid_controller_c)

# 基准测试（x86 主机上评估 PID 性能/数值指标）
add_executable(pid_bench bench/bench_pid.cpp)
target_link_libraries(pid_bench PRIVATE pid_lib pid_controller_c)
target_include_directories(pid_bench PRIVATE include)
install(TARGETS pid_bench RUNTIME DESTINATION bin)

# 安装
install(TARGETS rk_app RUNTIME DESTINATION bin)

# 测试配置（默认仅在非交叉编译时启用）。在受限网络下禁用自动拉取。
option(ENABLE_TESTS "Enable building tests" ON)
option(FETCH_GTEST "Fetch GoogleTest if not found" ON)
include(CTest)
if(BUILD_TESTING AND ENABLE_TESTS AND NOT CMAKE_CROSSCOMPILING)
  message(STATUS "Building tests for native host")
  find_package(GTest QUIET)
  if(NOT GTest_FOUND AND FETCH_GTEST)
    include(FetchContent)
    message(STATUS "GTest not found; fetching via FetchContent")
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  if(GTest_FOUND OR TARGET GTest::gtest_main)
    add_executable(pid_tests tests/test_pid.cpp)
    if(TARGET GTest::gtest_main)
      target_link_libraries(pid_tests PRIVATE GTest::gtest GTest::gtest_main pid_lib)
    else()
      target_link_libraries(pid_tests PRIVATE GTest::GTest GTest::Main pid_lib)
    endif()
    add_test(NAME pid_tests COMMAND pid_tests)
  else()
    message(STATUS "GTest not available; tests will not be built. Enable FETCH_GTEST to download.")
  endif()
endif()

# Object Detection Pipeline Libraries
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP yaml-cpp)
option(ENABLE_GIGE "Enable GigE Vision (Aravis) source" OFF)
if (ENABLE_GIGE)
  # Aravis（可选，用于未来直接使用Aravis API）；当前GigeSource仅依赖GStreamer
  pkg_check_modules(ARAVIS aravis-0.8)
  # GStreamer 与 appsink 为必需
  pkg_check_modules(GST REQUIRED gstreamer-1.0)
  pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)
endif()

# Core detection pipeline library
add_library(rkapp_core
    src/capture/FolderSource.cpp
    src/capture/VideoSource.cpp
    $<$<AND:$<BOOL:${ENABLE_GIGE}>,$<BOOL:${GST_FOUND}>>:src/capture/GigeSource.cpp>
    src/preprocess/Preprocess.cpp
    src/post/Postprocess.cpp
    src/output/TcpOutput.cpp
)
target_include_directories(rkapp_core PUBLIC include)
target_link_libraries(rkapp_core PUBLIC ${OpenCV_LIBS})
if(YAML_CPP_FOUND)
    target_link_libraries(rkapp_core PUBLIC ${YAML_CPP_LIBRARIES})
    target_include_directories(rkapp_core PUBLIC ${YAML_CPP_INCLUDE_DIRS})
endif()
if(ENABLE_GIGE AND GST_FOUND AND GST_APP_FOUND)
    target_compile_definitions(rkapp_core PUBLIC RKAPP_WITH_GIGE=1)
    target_link_libraries(rkapp_core PUBLIC ${GST_LIBRARIES} ${GST_APP_LIBRARIES})
    target_include_directories(rkapp_core PUBLIC ${GST_INCLUDE_DIRS} ${GST_APP_INCLUDE_DIRS})
    if(ARAVIS_FOUND)
        target_link_libraries(rkapp_core PUBLIC ${ARAVIS_LIBRARIES})
        target_include_directories(rkapp_core PUBLIC ${ARAVIS_INCLUDE_DIRS})
    endif()
endif()

# ONNX inference engine (optional)
if (ENABLE_ONNX)
    add_library(rkapp_infer_onnx src/infer/onnx/OnnxEngine.cpp)
    target_include_directories(rkapp_infer_onnx PUBLIC include)
    target_link_libraries(rkapp_infer_onnx PUBLIC ${OpenCV_LIBS} rkapp_core)

    # Prefer external ORT if ORT_HOME is set; fallback to third_party
    if(DEFINED ENV{ORT_HOME})
      message(STATUS "Using ORT from $ENV{ORT_HOME}")
      if(NOT EXISTS "$ENV{ORT_HOME}/include/onnxruntime_cxx_api.h")
        message(FATAL_ERROR "ORT_HOME is set but header not found: $ENV{ORT_HOME}/include/onnxruntime_cxx_api.h")
      endif()
      target_include_directories(rkapp_infer_onnx PRIVATE $ENV{ORT_HOME}/include)
      # Link by absolute path to avoid relying on link_directories
      find_library(ORT_LIB onnxruntime HINTS $ENV{ORT_HOME}/lib REQUIRED)
      target_link_libraries(rkapp_infer_onnx PUBLIC ${ORT_LIB})
      # Provider CUDA is optional; link if present, otherwise rely on runtime loading
      find_library(ORT_CUDA_PROVIDER onnxruntime_providers_cuda HINTS $ENV{ORT_HOME}/lib)
      if(ORT_CUDA_PROVIDER)
        target_link_libraries(rkapp_infer_onnx PUBLIC ${ORT_CUDA_PROVIDER})
      endif()
      target_compile_definitions(rkapp_infer_onnx PRIVATE ORT_WITH_CUDA=1)
    else()
      message(STATUS "Using bundled third_party ONNXRuntime (CPU)")
      target_include_directories(rkapp_infer_onnx PRIVATE third_party/onnxruntime/include)
      target_link_libraries(rkapp_infer_onnx PRIVATE ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib/libonnxruntime.so)
    endif()
    target_compile_definitions(rkapp_infer_onnx PUBLIC RKAPP_WITH_ONNX=1)
endif()

# RKNN inference engine (optional)
if (ENABLE_RKNN)
    add_library(rkapp_infer_rknn src/infer/rknn/RknnEngine.cpp)
    target_include_directories(rkapp_infer_rknn PUBLIC include)
    target_link_libraries(rkapp_infer_rknn PUBLIC ${OpenCV_LIBS})
    # RKNN SDK
    target_include_directories(rkapp_infer_rknn PRIVATE ${RKNN_HOME}/include)
    target_link_directories(rkapp_infer_rknn PRIVATE ${RKNN_HOME}/lib)
    target_link_libraries(rkapp_infer_rknn PRIVATE rknn_api)
    target_compile_definitions(rkapp_infer_rknn PRIVATE RKNN_PLATFORM=1)

    if (ENABLE_RGA)
      find_library(RGA_LIB rga)
      if (RGA_LIB)
        message(STATUS "Found RGA: ${RGA_LIB}")
        target_link_libraries(rkapp_infer_rknn PRIVATE ${RGA_LIB})
        target_compile_definitions(rkapp_infer_rknn PRIVATE RKNN_USE_RGA=1)
      else()
        message(STATUS "RGA not found; RKNN will fallback to OpenCV preprocess")
      endif()
    endif()
endif()

# Detection CLI application
add_executable(detect_cli examples/detect_cli.cpp)
target_link_libraries(detect_cli PRIVATE rkapp_core ${OpenCV_LIBS})
if(YAML_CPP_FOUND)
    target_link_libraries(detect_cli PRIVATE ${YAML_CPP_LIBRARIES})
endif()

if (ENABLE_ONNX)
    target_link_libraries(detect_cli PRIVATE rkapp_infer_onnx)
    target_compile_definitions(detect_cli PRIVATE RKAPP_WITH_ONNX=1)
endif()

if (ENABLE_RKNN)
    target_link_libraries(detect_cli PRIVATE rkapp_infer_rknn)
    target_compile_definitions(detect_cli PRIVATE RKAPP_WITH_RKNN=1)
endif()

# Install detection pipeline
install(TARGETS detect_cli RUNTIME DESTINATION bin)

# RKNN multicore sample (only built when RKNN enabled)
if (ENABLE_RKNN)
  add_executable(detect_rknn_multicore examples/detect_rknn_multicore.cpp)
  target_link_libraries(detect_rknn_multicore PRIVATE rkapp_infer_rknn ${OpenCV_LIBS})
  install(TARGETS detect_rknn_multicore RUNTIME DESTINATION bin)
endif()
