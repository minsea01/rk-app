/**
  ******************************************************************************
  * @file    pid_controller.h
  * @author  MINSEA
  * @version V1.0
  * @date    2024
  * @brief   通用PID控制器 - 替代FOC PID控制器
  ******************************************************************************
  */

#ifndef __PID_CONTROLLER_H
#define __PID_CONTROLLER_H

#ifdef __cplusplus
extern "C" {
#endif

/* 在嵌入式环境下可自行定义 STM32 头；桌面/主机环境下使用 stdint.h */
#ifdef STM32F4
#include "stm32f4xx.h"
#endif
#include <stdint.h>

/* PID Controller Structure */
typedef struct {
    float kp;               // 比例系数
    float ki;               // 积分系数
    float kd;               // 微分系数
    float beta;            // 2DOF 比例权重 (setpoint weight for P)
    float gamma;           // 2DOF 导数权重 (setpoint weight for D)
    float kd_lp_N;         // D项一阶滞后滤波器参数N（带宽因子）
    float backcalc_k;      // 积分背计算增益
    float setpoint;         // 设定值
    float output;           // 输出值
    float prev_error;       // 前一次误差（在2DOF路径中为 e_d）
    float last_e;          // 上一拍常规误差 e（用于统一对外查询）
    float integral;         // 积分累积
    float derivative;       // 微分值
    float prev_e_i;         // 上一次用于积分的误差（用于梯形积分）
    float output_min;       // 输出下限
    float output_max;       // 输出上限
    float integral_max;     // 积分限制
    uint32_t last_time;     // 上次计算时间
    
    // 兼容旧接口的成员
    float ki_internal;      // 内部积分系数
    float kd_internal;      // 内部微分系数
    float sample_time;      // 采样时间
    // 饱和状态共享
    float last_unsat;       // 最近一次未饱和输出
    float sat_ratio;        // 饱和比率
    
    /* --- Reference Governor (约束感知) --- */
    uint8_t rg_enabled;     // 参考治理器开关
    float rg_k;             // 参考治理器增益（设定值变化速率系数）
    float rg_sensitivity;   // 灵敏度近似S（用于margin归一化）
    float rg_setpoint;      // 形状化设定值（内部状态）
    float rg_max_rate;      // 设定值最大变化速率（可选）

    /* --- Disturbance Observer / ESO (扰动观测) --- */
    uint8_t eso_enabled;    // ESO开关
    float eso_x1;           // ESO状态1
    float eso_x2;           // ESO状态2
    float eso_beta1;        // ESO增益β1
    float eso_beta2;        // ESO增益β2
    float eso_b0;           // 被控对象名义增益 b0
    float eso_d_hat;        // 等效扰动估计

    /* --- 积分复位/泄放与输出限速 --- */
    uint8_t i_reset_enabled;    // 条件积分复位开关
    float i_reset_rho;          // 复位因子 0..1（<1表示部分复位）
    float i_leak_rate;          // 积分泄放率（1/s）
    float sat_time_accum;       // 饱和累计时间
    float sat_time_threshold;   // 饱和触发复位的阈值时间(s)

    uint8_t slew_enabled;   // 输出限速开关
    float slew_rate;        // 输出最大变化速率（单位：每秒）
    float prev_output;      // 上一拍输出（用于限速）

    // 其他内部状态
    float prev_input;       // 上一拍测量值（用于避免D项对设定值踢）
    uint8_t backcalc_enabled; // 背计算抗饱和开关

    /* --- 细化稳态与抗抖配置 --- */
    float i_deadband;       // 积分死区：|e| <= i_deadband 时暂停积分（0 关闭）
    float d_clip;           // D项限幅：|u_d| <= d_clip（0 关闭）

    /* --- 设定值滤波（2阶临界/过阻尼）与前馈 --- */
    uint8_t spf_enabled;    // 设定值滤波开关
    float spf_wn;           // 设定值滤波 自然频率(rad/s)
    float spf_zeta;         // 设定值滤波 阻尼比
    float spf_qf;           // 滤波后的设定值
    float spf_qfd;          // 滤波后的设定值一阶导
    float spf_qfdd;         // 滤波后的设定值二阶导
    float kff;              // 前馈增益（u_ff = kff * ref）
    float ff_k0, ff_k1, ff_k2; // 动态前馈：u_ff = k0*r + k1*dr + k2*d2r

    /* --- dt夹持配置（可关闭） --- */
    uint8_t dt_clamp_enabled; // 允许夹持（默认开启，沿用旧行为）
    float dt_min_factor;    // dt >= dt_min_factor * sample_time （默认 0.5）
    float dt_max_factor;    // dt <= min(dt_max_factor * sample_time, 0.05s) （默认 5.0）
} PID_Controller_t;

/* Function Prototypes */

void PID_Init(PID_Controller_t* pid, float kp, float ki, float kd, 
              float output_min, float output_max);

float PID_Update(PID_Controller_t* pid, float input);
float PID_Update_WithSetpoint(PID_Controller_t* pid, float input, float setpoint, float dt);
float PID_Update_2DOF(PID_Controller_t* pid, float input, float setpoint, float dt);

void PID_SetParameters(PID_Controller_t* pid, float kp, float ki, float kd);
void PID_Set2DOF(PID_Controller_t* pid, float beta, float gamma, float kd_lp_N);
void PID_SetBackcalc(PID_Controller_t* pid, float k_backcalc);
void PID_SetAntiWindupTt(PID_Controller_t* pid, float Tt);
void PID_SetSetpoint(PID_Controller_t* pid, float setpoint);
void PID_Reset(PID_Controller_t* pid);
void PID_SetOutputLimits(PID_Controller_t* pid, float min, float max);
void PID_SetIntegralLimit(PID_Controller_t* pid, float limit);
float PID_GetOutput(const PID_Controller_t* pid);
float PID_GetError(const PID_Controller_t* pid);
float PID_GetErrorDWeighted(const PID_Controller_t* pid);
void PID_SetSampleTime(PID_Controller_t* pid, float sample_time);
void PID_EnableWindupProtection(PID_Controller_t* pid, uint8_t enable);

/* Enhancements */
void PID_EnableReferenceGovernor(PID_Controller_t* pid, uint8_t enable);
void PID_ConfigReferenceGovernor(PID_Controller_t* pid, float k, float sensitivity, float max_rate);
void PID_EnableESO(PID_Controller_t* pid, uint8_t enable);
void PID_ConfigESO(PID_Controller_t* pid, float beta1, float beta2, float b0);
void PID_EnableIntegralReset(PID_Controller_t* pid, uint8_t enable);
void PID_ConfigIntegralReset(PID_Controller_t* pid, float rho, float leak_rate, float sat_time_threshold);
void PID_EnableSlewLimit(PID_Controller_t* pid, uint8_t enable);
void PID_ConfigSlewLimit(PID_Controller_t* pid, float slew_rate_per_sec);

/* Fine-tuning helpers */
void PID_SetIntegralDeadband(PID_Controller_t* pid, float deadband);
void PID_SetDerivativeClip(PID_Controller_t* pid, float d_clip_abs);

/* Setpoint filter & feedforward */
void PID_EnableSetpointFilter(PID_Controller_t* pid, uint8_t enable);
void PID_ConfigSetpointFilter(PID_Controller_t* pid, float wn, float zeta);
void PID_SetFeedforwardGain(PID_Controller_t* pid, float kff);
void PID_SetDynamicFeedforward(PID_Controller_t* pid, float k0, float k1, float k2);

/* dt clamp configuration */
void PID_ConfigDtClamp(PID_Controller_t* pid, uint8_t enable, float min_factor, float max_factor);

/* Legacy macro */
#define PID_Update_Legacy(pid, setpoint, input) \
    (PID_SetSetpoint(pid, setpoint), PID_Update(pid, input))

#ifdef __cplusplus
}
#endif

#endif /* __PID_CONTROLLER_H */

