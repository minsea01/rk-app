cmake_minimum_required(VERSION 3.16)
project(rk_app LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ARM64 (RK3588) 平台优化
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  message(STATUS "Configuring for ARM64 (RK3588) with Cortex-A76 optimizations")
  add_compile_options(
    -march=armv8.2-a+crypto+fp16  # RK3588 支持 ARMv8.2-A
    -mtune=cortex-a76             # 大核优化
  )
  add_compile_definitions(ARM_NEON=1)
  # Release-only aggressive optimizations (avoid fast-math in Debug)
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(
      -O3
      -ffast-math
      -ftree-vectorize
      -fomit-frame-pointer
    )
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()
endif()

# 导出 compile_commands.json 供 VS Code 智能感知
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 可选构建选项：Sanitizers 与 覆盖率（仅 x86 Debug 推荐）
option(ENABLE_SANITIZERS "Enable ASan/UBSan on x86 Debug" ON)
option(ENABLE_COVERAGE "Enable coverage flags on x86 Debug" OFF)

# Object detection pipeline options
option(ENABLE_ONNX "Enable ONNXRuntime engine" ON)
option(ENABLE_RKNN "Enable RKNN engine" OFF)
option(ENABLE_RGA "Enable RGA acceleration (if available)" ON)
option(ENABLE_MPP "Enable MPP hardware video decoding (RK3588)" OFF)
option(ENABLE_RKNN_DMA_FD "Enable RKNN DMA-BUF fd input (requires supported RKNN SDK)" OFF)
option(ENABLE_RKNN_IO_MEM "Enable RKNN IO memory for DMA-BUF zero-copy (RKNN >= 1.5.0)" OFF)
set(RKNN_HOME "/opt/rknpu2" CACHE PATH "RKNN SDK root")
set(MPP_HOME "/usr/include/rockchip" CACHE PATH "MPP SDK headers")
set(RKAPP_BUILD_WITH_ONNX OFF CACHE BOOL "Whether this build includes ONNXRuntime engine support" FORCE)

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

# 主程序 (已移除 rk_app 及其 PID 依赖)

# 安装
# install(TARGETS rk_app RUNTIME DESTINATION bin OPTIONAL)

# 测试配置（默认仅在非交叉编译时启用）。在受限网络下禁用自动拉取。
option(ENABLE_TESTS "Enable building tests" ON)
option(FETCH_GTEST "Fetch GoogleTest if not found" ON)
set(RKAPP_ENABLE_CPP_TESTS OFF)
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

  if(NOT (GTest_FOUND OR TARGET GTest::gtest_main))
    message(STATUS "Using builtin gtest-lite stub")
    add_library(gtest_stub INTERFACE)
    target_include_directories(gtest_stub INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gtest-lite/include)
    add_library(gtest_main_stub STATIC third_party/gtest-lite/src/gtest_main.cc)
    target_link_libraries(gtest_main_stub PUBLIC gtest_stub)
    add_library(GTest::gtest ALIAS gtest_stub)
    add_library(GTest::gtest_main ALIAS gtest_main_stub)
  endif()

  if(GTest_FOUND OR TARGET GTest::gtest_main)
    set(RKAPP_ENABLE_CPP_TESTS ON)
  else()
    message(STATUS "GTest not available; tests will not be built. Enable FETCH_GTEST to download.")
  endif()
endif()

# Object Detection Pipeline Libraries
# Add common Debian/Ubuntu multiarch OpenCV config paths so IDEs (no env) can find package
list(APPEND CMAKE_PREFIX_PATH
  /usr/lib/x86_64-linux-gnu/cmake/opencv4
  /usr/lib/aarch64-linux-gnu/cmake/opencv4
)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
set(HAVE_YAML_CPP OFF)
find_package(yaml-cpp QUIET)
if(yaml-cpp_FOUND)
  set(HAVE_YAML_CPP ON)
  if(TARGET yaml-cpp::yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
  else()
    set(YAML_CPP_TARGET yaml-cpp)
  endif()
else()
  pkg_check_modules(YAML_CPP yaml-cpp)
  if(YAML_CPP_FOUND)
    add_library(yaml-cpp::yaml-cpp INTERFACE IMPORTED)
    target_include_directories(yaml-cpp::yaml-cpp INTERFACE ${YAML_CPP_INCLUDE_DIRS})
    target_link_libraries(yaml-cpp::yaml-cpp INTERFACE ${YAML_CPP_LIBRARIES})
    set(HAVE_YAML_CPP ON)
    set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
  endif()
endif()

if(NOT HAVE_YAML_CPP)
  message(FATAL_ERROR "yaml-cpp not found. Install libyaml-cpp-dev or provide pkg-config.")
endif()
if(HAVE_YAML_CPP)
  add_compile_definitions(RKAPP_HAVE_YAMLCPP=1)
endif()
option(ENABLE_GIGE "Enable GigE Vision (Aravis) source" OFF)
option(ENABLE_CSI "Enable MIPI CSI camera source (v4l2src)" OFF)
if (ENABLE_GIGE OR ENABLE_CSI)
  # GStreamer 与 appsink 为必需
  pkg_check_modules(GST REQUIRED gstreamer-1.0)
  pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)
endif()
if (ENABLE_GIGE)
  # Aravis（可选，用于未来直接使用Aravis API）；当前GigeSource仅依赖GStreamer
  pkg_check_modules(ARAVIS aravis-0.8)
endif()

# Core detection pipeline library
add_library(rkapp_core
    src/capture/FolderSource.cpp
    src/capture/VideoSource.cpp
    src/capture/MppSource.cpp
    src/capture/GstSourceBase.cpp
    $<$<AND:$<BOOL:${ENABLE_GIGE}>,$<BOOL:${GST_FOUND}>>:src/capture/GigeSource.cpp>
    $<$<AND:$<BOOL:${ENABLE_CSI}>,$<BOOL:${GST_FOUND}>>:src/capture/CsiSource.cpp>
    src/preprocess/Preprocess.cpp
    src/post/Postprocess.cpp
    src/output/TcpOutput.cpp
    src/common/DmaBuf.cpp
    src/common/FramePool.cpp
)
target_include_directories(rkapp_core PUBLIC include)
target_link_libraries(rkapp_core PUBLIC ${OpenCV_LIBS})

set(RKAPP_RGA_FOUND OFF)
if (ENABLE_RGA)
    find_library(RGA_LIB NAMES rga)
    find_path(RGA_INCLUDE_DIR
        NAMES im2d.h rga.h
        PATH_SUFFIXES rga
    )
    if (RGA_LIB AND RGA_INCLUDE_DIR)
        message(STATUS "Found RGA: ${RGA_LIB}")
        message(STATUS "  RGA include: ${RGA_INCLUDE_DIR}")
        target_include_directories(rkapp_core PUBLIC ${RGA_INCLUDE_DIR})
        target_link_libraries(rkapp_core PUBLIC ${RGA_LIB})
        # Preprocess/MppSource are compiled in rkapp_core and gate RGA code on this macro.
        target_compile_definitions(rkapp_core PUBLIC RKNN_USE_RGA=1)
        set(RKAPP_RGA_FOUND ON)
    elseif(RGA_LIB)
        message(STATUS "RGA library found but headers are missing; rkapp_core will fallback to OpenCV preprocess/conversion")
    else()
        message(STATUS "RGA not found; rkapp_core will fallback to OpenCV preprocess/conversion")
    endif()
endif()

# OpenMP for parallel NMS (optional)
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(rkapp_core PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found - parallel NMS enabled")
endif()

# DRM/DMA-BUF support for zero-copy pipeline (ARM64 only)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    find_library(DRM_LIB drm)
    if(DRM_LIB)
        target_link_libraries(rkapp_core PUBLIC ${DRM_LIB})
        target_compile_definitions(rkapp_core PUBLIC RKAPP_WITH_DRM=1)
        message(STATUS "Found libdrm - DMA-BUF zero-copy enabled")
    endif()
endif()
if(HAVE_YAML_CPP)
    target_link_libraries(rkapp_core PUBLIC ${YAML_CPP_TARGET})
    if(YAML_CPP_INCLUDE_DIRS)
      target_include_directories(rkapp_core PUBLIC ${YAML_CPP_INCLUDE_DIRS})
    endif()
endif()
if((ENABLE_GIGE OR ENABLE_CSI) AND GST_FOUND AND GST_APP_FOUND)
    target_link_libraries(rkapp_core PUBLIC ${GST_LIBRARIES} ${GST_APP_LIBRARIES})
    target_include_directories(rkapp_core PUBLIC ${GST_INCLUDE_DIRS} ${GST_APP_INCLUDE_DIRS})
endif()

if(ENABLE_GIGE AND GST_FOUND AND GST_APP_FOUND)
    target_compile_definitions(rkapp_core PUBLIC RKAPP_WITH_GIGE=1)
    if(ARAVIS_FOUND)
        target_link_libraries(rkapp_core PUBLIC ${ARAVIS_LIBRARIES})
        target_include_directories(rkapp_core PUBLIC ${ARAVIS_INCLUDE_DIRS})
    endif()
endif()

if(ENABLE_CSI AND GST_FOUND AND GST_APP_FOUND)
    target_compile_definitions(rkapp_core PUBLIC RKAPP_WITH_CSI=1)
endif()

# MPP hardware video decoding (RK3588)
if (ENABLE_MPP)
    find_library(MPP_LIB rockchip_mpp HINTS /usr/lib/aarch64-linux-gnu)
    find_package(PkgConfig QUIET)
    if (PKG_CONFIG_FOUND)
        pkg_check_modules(AVFORMAT libavformat)
        pkg_check_modules(AVCODEC libavcodec)
        pkg_check_modules(AVUTIL libavutil)
    endif()

    if (MPP_LIB)
        message(STATUS "Found MPP: ${MPP_LIB}")
        target_compile_definitions(rkapp_core PUBLIC RKAPP_WITH_MPP=1)
        target_link_libraries(rkapp_core PUBLIC ${MPP_LIB})
        target_include_directories(rkapp_core PUBLIC ${MPP_HOME})

        # FFmpeg for demuxing (optional but recommended)
        if (AVFORMAT_FOUND AND AVCODEC_FOUND)
            message(STATUS "Found FFmpeg for MPP demuxing")
            target_link_libraries(rkapp_core PUBLIC
                ${AVFORMAT_LIBRARIES}
                ${AVCODEC_LIBRARIES}
                ${AVUTIL_LIBRARIES}
            )
            target_include_directories(rkapp_core PUBLIC
                ${AVFORMAT_INCLUDE_DIRS}
                ${AVCODEC_INCLUDE_DIRS}
            )
        else()
            message(WARNING "FFmpeg not found - MPP will have limited format support")
        endif()
    else()
        message(WARNING "MPP library not found - hardware video decoding disabled")
    endif()
endif()

# Shared decode utilities used by RKNN path and tests
add_library(rkapp_model_meta src/infer/ModelMeta.cpp)
target_include_directories(rkapp_model_meta PUBLIC include)

set(RKAPP_DECODE_UTILS_SOURCES
  src/infer/rknn/RknnDecodeUtils.cpp
  src/infer/rknn/RknnDecodeOptimized.cpp
)
add_library(rkapp_decode_utils ${RKAPP_DECODE_UTILS_SOURCES})
target_include_directories(rkapp_decode_utils PUBLIC include)
target_link_libraries(rkapp_decode_utils PUBLIC rkapp_model_meta)

if(RKAPP_ENABLE_CPP_TESTS)
  # C++ tests are in tests/cpp/CMakeLists.txt and depend on core libraries.
  add_subdirectory(tests/cpp)

  set(CORE_IO_TEST_SOURCES
    tests/unit/test_capture_gige.cpp
    tests/unit/test_output_tcp.cpp
  )
  if(NOT (ENABLE_GIGE AND GST_FOUND))
    list(APPEND CORE_IO_TEST_SOURCES src/capture/GigeSource.cpp)
  endif()

  add_executable(core_io_tests ${CORE_IO_TEST_SOURCES})
  target_include_directories(core_io_tests PRIVATE include)
  if(TARGET GTest::gtest_main)
    target_link_libraries(core_io_tests PRIVATE GTest::gtest GTest::gtest_main rkapp_core)
  else()
    target_link_libraries(core_io_tests PRIVATE GTest::GTest GTest::Main rkapp_core)
  endif()
  add_test(NAME core_io_tests COMMAND core_io_tests)
endif()

# ONNX inference engine (optional)
set(RKAPP_ONNX_INCLUDE_DIR "")
set(RKAPP_ONNX_EXTRA_LIBS "")
set(RKAPP_ONNX_PRIVATE_DEFS "")
if (ENABLE_ONNX)
    if(DEFINED ENV{ORT_HOME} AND NOT "$ENV{ORT_HOME}" STREQUAL "")
      message(STATUS "Using ORT from $ENV{ORT_HOME}")
      if(NOT EXISTS "$ENV{ORT_HOME}/include/onnxruntime_cxx_api.h")
        message(FATAL_ERROR "ORT_HOME is set but header not found: $ENV{ORT_HOME}/include/onnxruntime_cxx_api.h")
      endif()
      set(RKAPP_ONNX_INCLUDE_DIR "$ENV{ORT_HOME}/include")
      # Link by absolute path to avoid relying on link_directories.
      find_library(RKAPP_ONNX_LIBRARY onnxruntime HINTS "$ENV{ORT_HOME}/lib" REQUIRED)
      # Provider CUDA is optional; link if present, otherwise rely on runtime loading.
      find_library(RKAPP_ONNX_CUDA_PROVIDER onnxruntime_providers_cuda HINTS "$ENV{ORT_HOME}/lib")
      if(RKAPP_ONNX_CUDA_PROVIDER)
        list(APPEND RKAPP_ONNX_EXTRA_LIBS "${RKAPP_ONNX_CUDA_PROVIDER}")
        list(APPEND RKAPP_ONNX_PRIVATE_DEFS ORT_WITH_CUDA=1)
      endif()
      set(RKAPP_BUILD_WITH_ONNX ON CACHE BOOL "Whether this build includes ONNXRuntime engine support" FORCE)
    else()
      set(RKAPP_ONNX_BUNDLED_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
      set(RKAPP_ONNX_BUNDLED_INCLUDE "${RKAPP_ONNX_BUNDLED_ROOT}/include/onnxruntime_cxx_api.h")
      set(RKAPP_ONNX_BUNDLED_LIBRARY "${RKAPP_ONNX_BUNDLED_ROOT}/lib/libonnxruntime.so")
      if(EXISTS "${RKAPP_ONNX_BUNDLED_INCLUDE}" AND EXISTS "${RKAPP_ONNX_BUNDLED_LIBRARY}")
        message(STATUS "Using bundled third_party ONNXRuntime (CPU)")
        set(RKAPP_ONNX_INCLUDE_DIR "${RKAPP_ONNX_BUNDLED_ROOT}/include")
        set(RKAPP_ONNX_LIBRARY "${RKAPP_ONNX_BUNDLED_LIBRARY}")
        set(RKAPP_BUILD_WITH_ONNX ON CACHE BOOL "Whether this build includes ONNXRuntime engine support" FORCE)
      else()
        message(WARNING
          "ENABLE_ONNX=ON but ONNX Runtime was not found. "
          "Set ORT_HOME or populate third_party/onnxruntime to enable ONNX support. "
          "Continuing with ONNX support disabled for this build.")
      endif()
    endif()
endif()

if (RKAPP_BUILD_WITH_ONNX)
    set(RKAPP_ONNX_SOURCES
      src/infer/onnx/OnnxEngine.cpp
      src/infer/onnx/internal/OnnxEngineInternal.cpp
    )
    add_library(rkapp_infer_onnx ${RKAPP_ONNX_SOURCES})
    target_include_directories(rkapp_infer_onnx PUBLIC include)
    target_include_directories(rkapp_infer_onnx PRIVATE ${RKAPP_ONNX_INCLUDE_DIR})
    target_link_libraries(
      rkapp_infer_onnx
      PUBLIC
        ${OpenCV_LIBS}
        rkapp_core
        rkapp_decode_utils
        rkapp_model_meta
        ${RKAPP_ONNX_LIBRARY}
        ${RKAPP_ONNX_EXTRA_LIBS}
    )
    if(RKAPP_ONNX_PRIVATE_DEFS)
      target_compile_definitions(rkapp_infer_onnx PRIVATE ${RKAPP_ONNX_PRIVATE_DEFS})
    endif()
    target_compile_definitions(rkapp_infer_onnx PUBLIC RKAPP_WITH_ONNX=1)
endif()

# RKNN inference engine (optional)
if (ENABLE_RKNN)
    # Validate RKNN SDK installation
    if (NOT EXISTS "${RKNN_HOME}")
      message(FATAL_ERROR "RKNN_HOME='${RKNN_HOME}' does not exist.\n"
                          "Please set -DRKNN_HOME=/path/to/rknpu2 or install RKNN SDK to /opt/rknpu2")
    endif()
    if (NOT EXISTS "${RKNN_HOME}/include/rknn_api.h")
      message(FATAL_ERROR "RKNN SDK headers not found at ${RKNN_HOME}/include/rknn_api.h\n"
                          "Check RKNN_HOME path or SDK installation.")
    endif()
    set(RKNN_LIB_SEARCH_DIRS
        "${RKNN_HOME}/lib"
        "${RKNN_HOME}/lib64"
        "${RKNN_HOME}/aarch64"
    )
    find_library(RKNN_API_LIB NAMES rknn_api rknnrt PATHS ${RKNN_LIB_SEARCH_DIRS} NO_DEFAULT_PATH)
    if (NOT RKNN_API_LIB)
      message(FATAL_ERROR
        "RKNN library not found under ${RKNN_HOME}.\n"
        "Tried names: librknn_api.so, librknnrt.so\n"
        "Tried directories: ${RKNN_LIB_SEARCH_DIRS}\n"
        "Check RKNN SDK installation or RKNN_HOME path.")
    endif()
    message(STATUS "Found RKNN SDK: ${RKNN_HOME}")
    message(STATUS "  RKNN lib: ${RKNN_API_LIB}")

    set(RKAPP_RKNN_SOURCES
      src/infer/rknn/RknnEngine.cpp
      src/infer/rknn/internal/RknnEngineInternal.cpp
      src/infer/rknn/internal/RknnEngineDma.cpp
    )
    add_library(rkapp_infer_rknn ${RKAPP_RKNN_SOURCES})
    target_include_directories(rkapp_infer_rknn PUBLIC include)
    target_link_libraries(
      rkapp_infer_rknn
      PUBLIC
        ${OpenCV_LIBS}
        rkapp_core
        rkapp_decode_utils
        rkapp_model_meta
    )
    # RKNN SDK
    target_include_directories(rkapp_infer_rknn PRIVATE ${RKNN_HOME}/include)
    target_link_libraries(rkapp_infer_rknn PRIVATE ${RKNN_API_LIB})
    target_compile_definitions(rkapp_infer_rknn PRIVATE RKNN_PLATFORM=1)
    if (ENABLE_RKNN_DMA_FD)
      target_compile_definitions(rkapp_infer_rknn PRIVATE RKAPP_RKNN_DMA_FD_INPUT=1)
    endif()
    if (ENABLE_RKNN_IO_MEM)
      target_compile_definitions(rkapp_infer_rknn PRIVATE RKAPP_RKNN_IO_MEM=1)
    endif()

    if (ENABLE_RGA)
      if (RKAPP_RGA_FOUND)
        target_link_libraries(rkapp_infer_rknn PRIVATE ${RGA_LIB})
        target_compile_definitions(rkapp_infer_rknn PRIVATE RKNN_USE_RGA=1)
      else()
        message(STATUS "RGA not found; RKNN will fallback to OpenCV preprocess")
      endif()
    endif()
endif()

# High-performance detection pipeline (integrates all hardware acceleration)
add_library(rkapp_pipeline
    src/pipeline/BoxTracker.cpp
    src/pipeline/DetectionPipeline.cpp
    src/pipeline/PipelineConfigLoader.cpp
)
target_include_directories(rkapp_pipeline PUBLIC include)
target_link_libraries(rkapp_pipeline PUBLIC rkapp_core rkapp_model_meta)

if (ENABLE_RKNN)
    target_link_libraries(rkapp_pipeline PUBLIC rkapp_infer_rknn)
    target_compile_definitions(rkapp_pipeline PUBLIC RKAPP_WITH_RKNN=1)
endif()

if (RKAPP_BUILD_WITH_ONNX)
    target_link_libraries(rkapp_pipeline PUBLIC rkapp_infer_onnx)
    target_compile_definitions(rkapp_pipeline PUBLIC RKAPP_WITH_ONNX=1)
endif()

# Detection CLI application
add_executable(detect_cli examples/detect_cli.cpp)
target_link_libraries(detect_cli PRIVATE rkapp_pipeline ${OpenCV_LIBS})
if(HAVE_YAML_CPP)
    target_link_libraries(detect_cli PRIVATE ${YAML_CPP_TARGET})
endif()

# Install detection pipeline
install(TARGETS detect_cli RUNTIME DESTINATION bin)

# RKNN multicore sample (only built when RKNN enabled)
if (ENABLE_RKNN)
  add_executable(detect_rknn_multicore examples/detect_rknn_multicore.cpp)
  target_link_libraries(detect_rknn_multicore PRIVATE rkapp_infer_rknn ${OpenCV_LIBS} Threads::Threads)
  install(TARGETS detect_rknn_multicore RUNTIME DESTINATION bin)

  # High-performance pipeline demo
  add_executable(detect_pipeline examples/detect_pipeline.cpp)
  target_link_libraries(detect_pipeline PRIVATE rkapp_pipeline ${OpenCV_LIBS})
  install(TARGETS detect_pipeline RUNTIME DESTINATION bin)

  # End-to-end latency benchmark
  add_executable(bench_e2e_cpp examples/bench_e2e_cpp.cpp)
  target_link_libraries(bench_e2e_cpp PRIVATE rkapp_infer_rknn ${OpenCV_LIBS})
  target_compile_definitions(bench_e2e_cpp PRIVATE RKAPP_WITH_RKNN=1)
  install(TARGETS bench_e2e_cpp RUNTIME DESTINATION bin)
endif()
