#include "pid.h"

PID::PID(double kp, double ki, double kd) : kp_(kp), ki_(ki), kd_(kd) {}

void PID::setTunings(double kp, double ki, double kd) {
    kp_ = kp; ki_ = ki; kd_ = kd;
}

void PID::reset() {
    prev_error_ = 0.0;
    integral_ = 0.0;
}

double PID::compute(double setpoint, double measured) {
    const double error = setpoint - measured;
    integral_ += error;
    const double derivative = error - prev_error_;
    prev_error_ = error;
    return kp_ * error + ki_ * integral_ + kd_ * derivative;
}
