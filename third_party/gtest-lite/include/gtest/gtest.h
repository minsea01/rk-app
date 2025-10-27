#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace testing {

struct TestInfo {
    const char* case_name;
    const char* test_name;
    void (*func)();
};

inline std::vector<TestInfo>& registry() {
    static std::vector<TestInfo> tests;
    return tests;
}

class TestRegistrar {
public:
    TestRegistrar(const char* case_name, const char* test_name, void (*func)()) {
        registry().push_back({case_name, test_name, func});
    }
};

namespace detail {

inline int& failureCounter() {
    static int failures = 0;
    return failures;
}

inline bool*& currentTestFailedPtr() {
    static bool* ptr = nullptr;
    return ptr;
}

inline std::string toString(const std::string& value) {
    return value;
}

template <typename T>
std::string toString(const T& value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

inline void ReportFailure(const char* macro, const char* file, int line, const std::string& message) {
    std::cerr << file << ":" << line << ": " << macro << " failed: " << message << std::endl;
    failureCounter()++;
    if (currentTestFailedPtr()) {
        *currentTestFailedPtr() = true;
    }
}

inline bool ExpectCondition(bool condition,
                            const char* macro,
                            const char* file,
                            int line,
                            const std::string& message) {
    if (!condition) {
        ReportFailure(macro, file, line, message);
        return false;
    }
    return true;
}

template <typename A, typename B>
bool ExpectComparison(bool condition,
                      const A& actual,
                      const B& expected,
                      const char* expr_actual,
                      const char* expr_expected,
                      const char* macro,
                      const char* cmp,
                      const char* file,
                      int line) {
    if (condition) return true;
    std::ostringstream oss;
    oss << expr_actual << " " << cmp << " " << expr_expected
        << " (actual: " << toString(actual) << ", expected: " << toString(expected) << ")";
    ReportFailure(macro, file, line, oss.str());
    return false;
}

inline bool ExpectNear(double value,
                       double expected,
                       double abs_error,
                       const char* expr_value,
                       const char* expr_expected,
                       const char* macro,
                       const char* file,
                       int line) {
    if (std::fabs(value - expected) <= abs_error) return true;
    std::ostringstream oss;
    oss << expr_value << " ~= " << expr_expected
        << " (actual: " << value << ", expected: " << expected
        << ", abs error: " << std::fabs(value - expected)
        << ", tolerance: " << abs_error << ")";
    ReportFailure(macro, file, line, oss.str());
    return false;
}

}  // namespace detail

inline void InitGoogleTest(int*, char**) {}

inline int RUN_ALL_TESTS() {
    auto& tests = registry();
    std::cerr << "[==========] Running " << tests.size() << " tests from " << tests.size() << " test suites." << std::endl;
    int failed_tests = 0;
    for (const auto& test : tests) {
        std::cerr << "[ RUN      ] " << test.case_name << "." << test.test_name << std::endl;
        bool current_failed = false;
        detail::currentTestFailedPtr() = &current_failed;
        try {
            test.func();
        } catch (const std::exception& ex) {
            detail::ReportFailure("UNCAUGHT_EXCEPTION", __FILE__, __LINE__, ex.what());
            current_failed = true;
        } catch (...) {
            detail::ReportFailure("UNCAUGHT_EXCEPTION", __FILE__, __LINE__, "unknown exception");
            current_failed = true;
        }
        detail::currentTestFailedPtr() = nullptr;
        if (current_failed) {
            ++failed_tests;
            std::cerr << "[  FAILED  ] " << test.case_name << "." << test.test_name << std::endl;
        } else {
            std::cerr << "[       OK ] " << test.case_name << "." << test.test_name << std::endl;
        }
    }
    std::cerr << "[==========] " << tests.size() << " tests run. "
              << detail::failureCounter() << " assertions failed." << std::endl;
    return failed_tests == 0 ? 0 : 1;
}

namespace detail {

inline bool ExpectBool(bool value,
                       bool expected,
                       const char* expr,
                       const char* macro,
                       const char* file,
                       int line) {
    if (value == expected) return true;
    std::ostringstream oss;
    oss << expr << " is " << (value ? "true" : "false") << " (expected "
        << (expected ? "true" : "false") << ")";
    ReportFailure(macro, file, line, oss.str());
    return false;
}

}  // namespace detail

}  // namespace testing

#define TEST(SuiteName, TestName)                                                      \
    static void SuiteName##_##TestName##_Test();                                       \
    static ::testing::TestRegistrar SuiteName##_##TestName##_registrar(                \
        #SuiteName, #TestName, &SuiteName##_##TestName##_Test);                        \
    static void SuiteName##_##TestName##_Test()

#define EXPECT_TRUE(condition)                                                         \
    (void)::testing::detail::ExpectBool((condition), true, #condition, "EXPECT_TRUE",  \
                                        __FILE__, __LINE__)

#define EXPECT_FALSE(condition)                                                        \
    (void)::testing::detail::ExpectBool((condition), false, #condition, "EXPECT_FALSE",\
                                        __FILE__, __LINE__)

#define ASSERT_TRUE(condition)                                                         \
    do {                                                                               \
        if (!::testing::detail::ExpectBool((condition), true, #condition,              \
                                           "ASSERT_TRUE", __FILE__, __LINE__))         \
            return;                                                                    \
    } while (0)

#define ASSERT_FALSE(condition)                                                        \
    do {                                                                               \
        if (!::testing::detail::ExpectBool((condition), false, #condition,             \
                                           "ASSERT_FALSE", __FILE__, __LINE__))        \
            return;                                                                    \
    } while (0)

#define EXPECT_EQ(val1, val2)                                                          \
    do {                                                                               \
        const auto& _gtest_v1 = (val1);                                                \
        const auto& _gtest_v2 = (val2);                                                \
        (void)::testing::detail::ExpectComparison(_gtest_v1 == _gtest_v2,              \
            _gtest_v1, _gtest_v2, #val1, #val2, "EXPECT_EQ", "==",                     \
            __FILE__, __LINE__);                                                       \
    } while (0)

#define EXPECT_NE(val1, val2)                                                          \
    do {                                                                               \
        const auto& _gtest_v1 = (val1);                                                \
        const auto& _gtest_v2 = (val2);                                                \
        (void)::testing::detail::ExpectComparison(_gtest_v1 != _gtest_v2,              \
            _gtest_v1, _gtest_v2, #val1, #val2, "EXPECT_NE", "!=",                     \
            __FILE__, __LINE__);                                                       \
    } while (0)

#define EXPECT_GT(val1, val2)                                                          \
    do {                                                                               \
        const auto& _gtest_v1 = (val1);                                                \
        const auto& _gtest_v2 = (val2);                                                \
        (void)::testing::detail::ExpectComparison(_gtest_v1 > _gtest_v2,               \
            _gtest_v1, _gtest_v2, #val1, #val2, "EXPECT_GT", ">",                      \
            __FILE__, __LINE__);                                                       \
    } while (0)

#define EXPECT_GE(val1, val2)                                                          \
    do {                                                                               \
        const auto& _gtest_v1 = (val1);                                                \
        const auto& _gtest_v2 = (val2);                                                \
        (void)::testing::detail::ExpectComparison(_gtest_v1 >= _gtest_v2,              \
            _gtest_v1, _gtest_v2, #val1, #val2, "EXPECT_GE", ">=",                     \
            __FILE__, __LINE__);                                                       \
    } while (0)

#define EXPECT_NEAR(val1, val2, abs_error)                                             \
    (void)::testing::detail::ExpectNear(static_cast<double>(val1),                     \
                                        static_cast<double>(val2),                     \
                                        static_cast<double>(abs_error),                \
                                        #val1, #val2, "EXPECT_NEAR",                   \
                                        __FILE__, __LINE__)

#define ASSERT_EQ(val1, val2)                                                          \
    do {                                                                               \
        const auto& _gtest_v1 = (val1);                                                \
        const auto& _gtest_v2 = (val2);                                                \
        if (!::testing::detail::ExpectComparison(_gtest_v1 == _gtest_v2,               \
            _gtest_v1, _gtest_v2, #val1, #val2, "ASSERT_EQ", "==",                     \
            __FILE__, __LINE__))                                                       \
            return;                                                                    \
    } while (0)

#define ASSERT_NE(val1, val2)                                                          \
    do {                                                                               \
        const auto& _gtest_v1 = (val1);                                                \
        const auto& _gtest_v2 = (val2);                                                \
        if (!::testing::detail::ExpectComparison(_gtest_v1 != _gtest_v2,               \
            _gtest_v1, _gtest_v2, #val1, #val2, "ASSERT_NE", "!=",                     \
            __FILE__, __LINE__))                                                       \
            return;                                                                    \
    } while (0)

#define EXPECT_LT(val1, val2)                                                          \
    do {                                                                               \
        const auto& _gtest_v1 = (val1);                                                \
        const auto& _gtest_v2 = (val2);                                                \
        (void)::testing::detail::ExpectComparison(_gtest_v1 < _gtest_v2,               \
            _gtest_v1, _gtest_v2, #val1, #val2, "EXPECT_LT", "<",                      \
            __FILE__, __LINE__);                                                       \
    } while (0)
