# C++ Unit Tests with Google Test

cmake_minimum_required(VERSION 3.16)

# Enable testing
enable_testing()

# Find Google Test
find_package(GTest)

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
