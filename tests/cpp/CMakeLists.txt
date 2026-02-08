# C++ Unit Tests with Google Test

cmake_minimum_required(VERSION 3.16)

# Enable testing
enable_testing()

# Find required packages
find_package(GTest)
find_package(OpenCV REQUIRED)

if(NOT GTest_FOUND)
    message(STATUS "Google Test not found, fetching from GitHub...")

    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )

    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)
endif()

# Test executable for preprocessing
add_executable(test_preprocess test_preprocess.cpp)

target_link_libraries(test_preprocess
    PRIVATE
        GTest::gtest
        GTest::gtest_main
)

# Add test to CTest
add_test(NAME PreprocessTests COMMAND test_preprocess)

# Set test properties
set_tests_properties(PreprocessTests PROPERTIES
    TIMEOUT 30
    LABELS "unit"
)

# Code coverage support (if enabled)
if(ENABLE_COVERAGE)
    target_compile_options(test_preprocess PRIVATE --coverage)
    target_link_options(test_preprocess PRIVATE --coverage)
endif()

# Test executable for RKNN decode utilities
add_executable(test_rknn_decode test_rknn_decode.cpp)

target_link_libraries(test_rknn_decode
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        rkapp_decode_utils
)

add_test(NAME RknnDecodeTests COMMAND test_rknn_decode)

set_tests_properties(RknnDecodeTests PROPERTIES
    TIMEOUT 30
    LABELS "unit"
)

if(ENABLE_COVERAGE)
    target_compile_options(test_rknn_decode PRIVATE --coverage)
    target_link_options(test_rknn_decode PRIVATE --coverage)
endif()

# Test executable for RKNN inference consistency
add_executable(test_rknn_infer_consistency test_rknn_infer_consistency.cpp)

target_link_libraries(test_rknn_infer_consistency
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        ${OpenCV_LIBS}
)

add_test(NAME RknnInferConsistencyTests COMMAND test_rknn_infer_consistency)

set_tests_properties(RknnInferConsistencyTests PROPERTIES
    TIMEOUT 30
    LABELS "unit"
)

if(ENABLE_COVERAGE)
    target_compile_options(test_rknn_infer_consistency PRIVATE --coverage)
    target_link_options(test_rknn_infer_consistency PRIVATE --coverage)
endif()

# Test executable for DMA-BUF helpers
add_executable(test_dmabuf test_dmabuf.cpp)

target_link_libraries(test_dmabuf
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        rkapp_core
        ${OpenCV_LIBS}
)

add_test(NAME DmaBufTests COMMAND test_dmabuf)

set_tests_properties(DmaBufTests PROPERTIES
    TIMEOUT 30
    LABELS "unit"
)

if(ENABLE_COVERAGE)
    target_compile_options(test_dmabuf PRIVATE --coverage)
    target_link_options(test_dmabuf PRIVATE --coverage)
endif()

# Test executable for detection pipeline factory helpers
add_executable(test_pipeline_factory test_pipeline_factory.cpp)

target_link_libraries(test_pipeline_factory
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        rkapp_pipeline
)

add_test(NAME DetectionPipelineFactoryTests COMMAND test_pipeline_factory)

set_tests_properties(DetectionPipelineFactoryTests PROPERTIES
    TIMEOUT 30
    LABELS "unit"
)

if(ENABLE_COVERAGE)
    target_compile_options(test_pipeline_factory PRIVATE --coverage)
    target_link_options(test_pipeline_factory PRIVATE --coverage)
endif()
