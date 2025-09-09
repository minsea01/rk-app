/**
  ******************************************************************************
  * @file    pid_controller.c
  * @author  MINSEA
  * @version V1.0
  * @date    2024
  * @brief   通用PID控制器实现
  ******************************************************************************
  */

#include "pid_controller.h"
#include <math.h>
#include "drivers/time_service.h"
#include <stdbool.h>
#include <stddef.h>

#define PID_EPSILON     1e-6f       // 防除零保护

static inline float _imax_from_limits(float ki, float umin, float umax){
    if (fabsf(ki) < 1e-6f) return 0.0f;         // ki==0 冻结积分
    return (umax - umin) / fabsf(ki);
}

static inline float _clampf(float x, float lo, float hi){
    return x < lo ? lo : (x > hi ? hi : x);
}

void PID_Init(PID_Controller_t* pid, float kp, float ki, float kd, 
              float output_min, float output_max)
{
    if (pid == NULL) return;
    
    pid->kp = kp;
    pid->ki = ki;
    pid->kd = kd;
    pid->beta = 1.0f;
    pid->gamma = 0.0f;
    pid->kd_lp_N = 20.0f;
    pid->backcalc_k = ki;
    pid->setpoint = 0.0f;
    pid->output = 0.0f;
    pid->prev_error = 0.0f;
    pid->integral = 0.0f;
    pid->derivative = 0.0f;
    pid->prev_e_i = 0.0f;
    pid->output_min = output_min;
    pid->output_max = output_max;
    pid->integral_max = _imax_from_limits(ki, output_min, output_max);
    pid->last_time = micros();
    
    // 初始化兼容成员
    pid->ki_internal = ki;
    pid->kd_internal = kd;
    pid->sample_time = 0.001f; // 默认1ms
    pid->last_unsat = 0.0f;
    pid->sat_ratio = 0.0f;

    // Enhancements defaults
    pid->rg_enabled = 0;
    pid->rg_k = 0.0f;
    pid->rg_sensitivity = 1.0f;
    pid->rg_setpoint = 0.0f;
    pid->rg_max_rate = 0.0f;

    pid->eso_enabled = 0;
    pid->eso_x1 = 0.0f;
    pid->eso_x2 = 0.0f;
    pid->eso_beta1 = 0.0f;
    pid->eso_beta2 = 0.0f;
    pid->eso_b0 = 1.0f;
    pid->eso_d_hat = 0.0f;

    pid->i_reset_enabled = 0;
    pid->i_reset_rho = 0.5f;
    pid->i_leak_rate = 0.0f;
    pid->sat_time_accum = 0.0f;
    pid->sat_time_threshold = 0.1f;

    pid->slew_enabled = 0;
    pid->slew_rate = 0.0f; // 0 = disabled
    pid->prev_output = 0.0f;
    pid->prev_input = 0.0f;
    pid->backcalc_enabled = 1; // 默认开启背计算抗饱和

    // 细化稳态与抗抖默认关闭
    pid->i_deadband = 0.0f;
    pid->d_clip = 0.0f;

    // 设定值滤波与前馈默认关闭
    pid->spf_enabled = 0;
    pid->spf_wn = 0.0f;
    pid->spf_zeta = 1.0f;
    pid->spf_qf = 0.0f;
    pid->spf_qfd = 0.0f;
    pid->spf_qfdd = 0.0f;
    pid->kff = 0.0f;
    pid->ff_k0 = 0.0f;
    pid->ff_k1 = 0.0f;
    pid->ff_k2 = 0.0f;

    // dt 夹持参数（保持与旧行为一致）
    pid->dt_clamp_enabled = 1;
    pid->dt_min_factor = 0.5f;
    pid->dt_max_factor = 5.0f;
}

float PID_Update(PID_Controller_t* pid, float input)
{
    if (pid == NULL) return 0.0f;
    uint32_t now = micros();
    uint32_t dt_us = now - pid->last_time;   // 允许 32bit 回绕
    if (dt_us == 0) return pid->output;
    pid->last_time = now;
    float dt = dt_us * 1e-6f;
    // clamp dt：防止调度抖动造成的大步长/小步长（可配置）
    if (pid->dt_clamp_enabled) {
        float dt_max = fminf(pid->dt_max_factor * pid->sample_time, 0.05f);
        float dt_min = pid->dt_min_factor * pid->sample_time;
        if (dt > dt_max) dt = dt_max;
        if (dt < dt_min) dt = dt_min;
    }
    return PID_Update_2DOF(pid, input, pid->setpoint, dt);
}

void PID_SetParameters(PID_Controller_t* pid, float kp, float ki, float kd)
{
    if (pid == NULL) return;
    
    pid->kp = kp;
    pid->ki = ki;
    pid->kd = kd;
    pid->backcalc_k = pid->ki;
    pid->integral_max = _imax_from_limits(pid->ki, pid->output_min, pid->output_max);
    pid->integral     = _clampf(pid->integral, -pid->integral_max, pid->integral_max);
    pid->prev_e_i     = 0.0f;
}

void PID_SetSetpoint(PID_Controller_t* pid, float setpoint)
{
    if (pid == NULL) return;
    
    pid->setpoint = setpoint;
}

void PID_Reset(PID_Controller_t* pid)
{
    if (pid == NULL) return;
    
    pid->prev_error = 0.0f;
    pid->integral = 0.0f;               
    pid->derivative = 0.0f;
    pid->prev_e_i = 0.0f;
    pid->output = 0.0f;
    pid->prev_output = 0.0f;
    pid->prev_input = 0.0f;
    pid->sat_time_accum = 0.0f;
    pid->rg_setpoint = pid->setpoint;
    pid->eso_x1 = 0.0f;
    pid->eso_x2 = 0.0f;
    pid->eso_d_hat = 0.0f;
    pid->last_time = micros();
}

void PID_SetOutputLimits(PID_Controller_t* pid, float min, float max)
{
    if (pid == NULL || min >= max) return;
    
    pid->output_min = min;
    pid->output_max = max;
    pid->integral_max = _imax_from_limits(pid->ki, min, max);
    pid->integral     = _clampf(pid->integral, -pid->integral_max, pid->integral_max);
}

void PID_SetIntegralLimit(PID_Controller_t* pid, float limit)
{
    if (pid == NULL || limit < 0.0f) return;
    pid->integral_max = limit;
    pid->integral     = _clampf(pid->integral, -pid->integral_max, pid->integral_max);
}

float PID_GetOutput(const PID_Controller_t* pid)
{
    if (pid == NULL) return 0.0f;
    return pid->output;
}

float PID_GetError(const PID_Controller_t* pid)
{
    return pid ? pid->last_e : 0.0f;
}

float PID_Update_WithSetpoint(PID_Controller_t* pid, float input, float setpoint, float dt)
{
    if (pid == NULL) return 0.0f;
    
    // 设置设定值
    pid->setpoint = setpoint;
    
    // 计算误差
    float error = setpoint - input;
    pid->last_e = error;
    
    // 比例项
    float proportional = pid->kp * error;
    
    // 积分项 (使用固定dt，带死区)
    if (fabsf(error) > pid->i_deadband) {
        pid->integral += error * dt;
    }
    
    // 积分限制 (使用辅助函数)
    pid->integral = _clampf(pid->integral, -pid->integral_max, pid->integral_max);

    float integral = pid->ki * pid->integral;
    
    // 微分项：对测量值微分并低通，避免设定值踢
    float dm = (dt > PID_EPSILON) ? (input - pid->prev_input) / dt : 0.0f;
    float N = fmaxf(1.0f, pid->kd_lp_N);
    float alpha = (N * dt) / (1.0f + N * dt);
    pid->derivative += alpha * ((-dm) - pid->derivative);
    float differential = pid->kd * pid->derivative;
    if (pid->d_clip > 0.0f) {
        if (differential >  pid->d_clip) differential =  pid->d_clip;
        if (differential < -pid->d_clip) differential = -pid->d_clip;
    }
    
    // 计算输出
    float u_unsat = proportional + integral + differential;
    pid->last_unsat = u_unsat;
    pid->output = u_unsat;
    
    // 输出限制
    if (pid->output > pid->output_max) {
        pid->output = pid->output_max;
    } else if (pid->output < pid->output_min) {
        pid->output = pid->output_min;
    }

    // 饱和比率与背计算积分抗饱和（遵循开关）
    pid->sat_ratio = (fabsf(u_unsat) > PID_EPSILON) ? fmaxf(-1.0f, fminf(1.0f, pid->output / u_unsat)) : 0.0f;
    if (pid->backcalc_enabled && fabsf(pid->ki) > 1e-6f) {
        float backcalc = pid->backcalc_k * (pid->output - u_unsat);
        pid->integral += backcalc * dt / pid->ki;
        pid->integral = _clampf(pid->integral, -pid->integral_max, pid->integral_max);
    }
    
    // 更新状态
    pid->prev_error = error;
    pid->prev_input = input;
    
    return pid->output;
}

void PID_SetSampleTime(PID_Controller_t* pid, float sample_time)
{
    if (pid == NULL) return;
    pid->sample_time = sample_time;
}

void PID_EnableWindupProtection(PID_Controller_t* pid, uint8_t enable)
{
    if (pid == NULL) return;
    pid->backcalc_enabled = enable ? 1 : 0;
}

void PID_EnableReferenceGovernor(PID_Controller_t* pid, uint8_t enable)
{
    if (!pid) return;
    pid->rg_enabled = enable ? 1 : 0;
}

void PID_ConfigReferenceGovernor(PID_Controller_t* pid, float k, float sensitivity, float max_rate)
{
    if (!pid) return;
    pid->rg_k = k;
    pid->rg_sensitivity = fmaxf(1e-3f, sensitivity);
    pid->rg_max_rate = fmaxf(0.0f, max_rate);
}

void PID_EnableESO(PID_Controller_t* pid, uint8_t enable)
{
    if (!pid) return;
    pid->eso_enabled = enable ? 1 : 0;
}

void PID_ConfigESO(PID_Controller_t* pid, float beta1, float beta2, float b0)
{
    if (!pid) return;
    pid->eso_beta1 = beta1;
    pid->eso_beta2 = beta2;
    pid->eso_b0 = (fabsf(b0) < 1e-6f) ? 1.0f : b0;
}

void PID_EnableIntegralReset(PID_Controller_t* pid, uint8_t enable)
{
    if (!pid) return;
    pid->i_reset_enabled = enable ? 1 : 0;
}

void PID_ConfigIntegralReset(PID_Controller_t* pid, float rho, float leak_rate, float sat_time_threshold)
{
    if (!pid) return;
    pid->i_reset_rho = _clampf(rho, 0.0f, 1.0f);
    pid->i_leak_rate = fmaxf(0.0f, leak_rate);
    pid->sat_time_accum = 0.0f;
    pid->sat_time_threshold = fmaxf(0.0f, sat_time_threshold);
}

void PID_EnableSlewLimit(PID_Controller_t* pid, uint8_t enable)
{
    if (!pid) return;
    pid->slew_enabled = enable ? 1 : 0;
}

void PID_ConfigSlewLimit(PID_Controller_t* pid, float slew_rate_per_sec)
{
    if (!pid) return;
    pid->slew_rate = fmaxf(0.0f, slew_rate_per_sec);
}

float PID_Update_2DOF(PID_Controller_t* pid, float input, float setpoint, float dt)
{
    if (pid == NULL) return 0.0f;
    // Reference Governor: shape setpoint before computing errors
    float setpoint_rg = setpoint;
    if (pid->rg_enabled) {
        if (pid->rg_setpoint == 0.0f && pid->setpoint != 0.0f) {
            pid->rg_setpoint = pid->setpoint;
        }
        float dr = setpoint - pid->rg_setpoint; // 设定值变化方向
        float e_p_pred = pid->beta * pid->rg_setpoint - input;
        float u_pred = pid->kp * e_p_pred + ((fabsf(pid->ki) > PID_EPSILON) ? pid->ki * pid->integral : 0.0f) + pid->kd * pid->derivative;
        float sens = fmaxf(pid->rg_sensitivity, 1e-3f);
        float margin_hi = pid->output_max - u_pred;
        float margin_lo = u_pred - pid->output_min;
        float margin_dir = (dr >= 0.0f) ? margin_hi : margin_lo;
        float rate = pid->rg_k * fmaxf(0.0f, margin_dir / sens);
        if (pid->rg_max_rate > 0.0f && rate > pid->rg_max_rate) rate = pid->rg_max_rate;
        float max_step = rate * dt;
        if (dr >  max_step) dr =  max_step;
        if (dr < -max_step) dr = -max_step;
        pid->rg_setpoint += dr;
        setpoint_rg = pid->rg_setpoint;
    }

    // 设定值滤波（二阶），位于 RG 之后
    float setpoint_f = setpoint_rg;
    if (pid->spf_enabled && pid->spf_wn > 1e-6f) {
        float wn = pid->spf_wn;
        float zeta = fmaxf(0.1f, pid->spf_zeta);
        float qfdd = wn*wn*(setpoint_rg - pid->spf_qf) - 2.0f*zeta*wn*pid->spf_qfd;
        pid->spf_qfdd = qfdd;
        pid->spf_qfd += pid->spf_qfdd * dt;
        pid->spf_qf  += pid->spf_qfd * dt;
        setpoint_f = pid->spf_qf;
    }

    pid->setpoint = setpoint_f;
    
    float e = setpoint_f - input;
    pid->last_e = e;
    float e_p = pid->beta * setpoint_f - input;   // P的设定权重
    float e_d = pid->gamma * setpoint_f - input;  // D的设定权重
    
    // 比例
    float up = pid->kp * e_p;
    
    // 条件积分 + 梯形积分（I 使用“真误差” e）
    bool enable_i = fabsf(pid->ki) > 1e-6f;
    bool at_hi = false, at_lo = false;
    {
        float u_pred = pid->kp * e_p + (enable_i ? pid->ki * pid->integral : 0.0f) + pid->kd * pid->derivative;
        at_hi = (u_pred > pid->output_max);
        at_lo = (u_pred < pid->output_min);
    }
    bool pushing_out = (at_hi && e > 0.0f) || (at_lo && e < 0.0f);
    if (enable_i && !pushing_out && fabsf(e) > pid->i_deadband) {
        float e_i = e;
        pid->integral += 0.5f * (e_i + pid->prev_e_i) * dt;
        pid->integral = _clampf(pid->integral, -pid->integral_max, pid->integral_max);
        pid->prev_e_i = e_i;
    }
    float ui = enable_i ? (pid->ki * pid->integral) : 0.0f;
    
    // 微分（带标准离散低通）
    float de = (dt > PID_EPSILON) ? (e_d - pid->prev_error) / dt : 0.0f;
    float N = fmaxf(1.0f, pid->kd_lp_N);
    float alpha = (N * dt) / (1.0f + N * dt);
    pid->derivative += alpha * (de - pid->derivative);
    float ud = pid->kd * pid->derivative;
    if (pid->d_clip > 0.0f) {
        if (ud >  pid->d_clip) ud =  pid->d_clip;
        if (ud < -pid->d_clip) ud = -pid->d_clip;
    }
    
    // 未饱和输出
    float u_ff = pid->kff * setpoint_f + pid->ff_k0 * setpoint_f + pid->ff_k1 * pid->spf_qfd + pid->ff_k2 * pid->spf_qfdd;
    float u_unsat = up + ui + ud + u_ff;
    pid->last_unsat = u_unsat;
    
    // 扰动补偿：对未饱和输出先行补偿，再统一限幅与背计算
    float u_comp = u_unsat;
    if (pid->eso_enabled && fabsf(pid->eso_b0) > 1e-6f) {
        u_comp = u_unsat - pid->eso_d_hat / pid->eso_b0; // 使用上一拍扰动估计
    }
    // 限幅 -> 限速，得到最终执行量
    float u = u_comp;
    if (u > pid->output_max) u = pid->output_max;
    if (u < pid->output_min) u = pid->output_min;
    pid->output = u;

    // 输出限速（slew rate）提前到此，确保后续使用最终输出
    if (pid->slew_enabled && pid->slew_rate > 0.0f) {
        float max_delta = pid->slew_rate * dt;
        float delta = pid->output - pid->prev_output;
        if (delta >  max_delta) delta =  max_delta;
        if (delta < -max_delta) delta = -max_delta;
        pid->output = pid->prev_output + delta;
    }

    // 饱和比率与背计算（基于最终输出）
    pid->sat_ratio = (fabsf(u_comp) > PID_EPSILON) ? fmaxf(-1.0f, fminf(1.0f, pid->output / u_comp)) : 0.0f;
    if (pid->backcalc_enabled && enable_i) {
        float backcalc = pid->backcalc_k * (pid->output - u_comp);
        pid->integral += backcalc * dt / pid->ki;  // enable_i 保证 ki 非零
        pid->integral = _clampf(pid->integral, -pid->integral_max, pid->integral_max);
    }

    // 饱和累计时间与积分复位/泄放（基于最终输出）
    if (fabsf(pid->output - u_unsat) > 1e-6f) {
        pid->sat_time_accum += dt;
    } else {
        pid->sat_time_accum = 0.0f;
    }
    if (pid->i_reset_enabled) {
        if (pid->sat_time_accum > pid->sat_time_threshold) {
            pid->integral *= _clampf(pid->i_reset_rho, 0.0f, 1.0f);
            pid->sat_time_accum = 0.0f;
        }
        if (pid->i_leak_rate > 0.0f) {
            float leak = 1.0f - _clampf(pid->i_leak_rate * dt, 0.0f, 0.5f);
            pid->integral *= leak;
        }
    }

    // 扰动观测器更新（使用最终执行的控制量）
    if (pid->eso_enabled && fabsf(pid->eso_b0) > 1e-6f) {
        float e_obs = input - pid->eso_x1;
        pid->eso_x1 += dt * (pid->eso_x2 + pid->eso_beta1 * e_obs);
        pid->eso_x2 += dt * (pid->eso_beta2 * e_obs + pid->eso_b0 * pid->output);
        pid->eso_d_hat = pid->eso_x2;
    }
    pid->prev_output = pid->output;
    
    // 更新
    pid->prev_error = e_d;
    pid->prev_input = input;
    return pid->output;
}

void PID_SetIntegralDeadband(PID_Controller_t* pid, float deadband)
{
    if (!pid) return;
    pid->i_deadband = fmaxf(0.0f, deadband);
}

void PID_SetDerivativeClip(PID_Controller_t* pid, float d_clip_abs)
{
    if (!pid) return;
    pid->d_clip = fmaxf(0.0f, d_clip_abs);
}

void PID_EnableSetpointFilter(PID_Controller_t* pid, uint8_t enable)
{
    if (!pid) return;
    pid->spf_enabled = enable ? 1 : 0;
}

void PID_ConfigSetpointFilter(PID_Controller_t* pid, float wn, float zeta)
{
    if (!pid) return;
    pid->spf_wn = fmaxf(0.0f, wn);
    pid->spf_zeta = zeta;
    pid->spf_qf = 0.0f;
    pid->spf_qfd = 0.0f;
}

void PID_SetFeedforwardGain(PID_Controller_t* pid, float kff)
{
    if (!pid) return;
    pid->kff = kff;
}

void PID_SetDynamicFeedforward(PID_Controller_t* pid, float k0, float k1, float k2)
{
    if (!pid) return;
    pid->ff_k0 = k0;
    pid->ff_k1 = k1;
    pid->ff_k2 = k2;
}

void PID_ConfigDtClamp(PID_Controller_t* pid, uint8_t enable, float min_factor, float max_factor)
{
    if (!pid) return;
    pid->dt_clamp_enabled = enable ? 1 : 0;
    pid->dt_min_factor = fmaxf(0.0f, min_factor);
    pid->dt_max_factor = fmaxf(1.0f, max_factor);
}

void PID_Set2DOF(PID_Controller_t* pid, float beta, float gamma, float kd_lp_N)
{
    if (pid == NULL) return;
    pid->beta = beta;
    pid->gamma = gamma;
    pid->kd_lp_N = (kd_lp_N < 1.0f) ? 1.0f : kd_lp_N;
}

void PID_SetBackcalc(PID_Controller_t* pid, float k_backcalc)
{
    if (pid == NULL) return;
    pid->backcalc_k = k_backcalc;
}

void PID_SetAntiWindupTt(PID_Controller_t* pid, float Tt)
{
    if (pid == NULL || Tt <= 0.0f) return;
    pid->backcalc_k = pid->ki / Tt;
}

float PID_GetErrorDWeighted(const PID_Controller_t* pid)
{
    return pid ? pid->prev_error : 0.0f;
}
