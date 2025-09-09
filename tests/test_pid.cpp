#include <gtest/gtest.h>
#include "pid.h"

TEST(PIDTest, Basic) {
    PID pid(1.0, 0.0, 0.0);
    EXPECT_NEAR(pid.compute(10, 0), 10.0, 1e-5);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
