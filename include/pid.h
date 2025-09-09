#pragma once

class PID {
public:
    PID(double kp, double ki, double kd);

    // 计算控制量：输入目标值与测量值
    double compute(double setpoint, double measured);

    // 重置内部状态
    void reset();

    // 调整参数
    void setTunings(double kp, double ki, double kd);

private:
    double kp_{};
    double ki_{};
    double kd_{};

    double prev_error_{};
    double integral_{};
};
