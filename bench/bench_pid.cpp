// Simple PID performance/response benchmark
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "pid.h"                // C++ PID
extern "C" {
#include "pid_controller.h"     // C PID
}

struct Args {
  std::string impl = "cpp";      // cpp | c
  std::string plant = "second";  // first | second
  int steps = 5000;
  double dt = 0.001;              // seconds
  double sp = 1.0;                // step size
  // First-order plant: dy/dt = (-y + K*u)/tau
  double tau = 0.05;              // 50 ms time constant
  double K = 1.0;
  // Second-order: x'' + 2*zeta*wn x' + wn^2 x = K*u
  double zeta = 0.7;
  double wn = 20.0;               // rad/s
  // PID gains
  double kp = 1.0, ki = 0.0, kd = 0.0;
  // 2DOF weights (C impl only)
  double beta = 1.0;              // P setpoint weight
  double gamma = 0.0;             // D setpoint weight
  double kdN = 20.0;              // D lowpass N
  // Feedforward and setpoint filter (C impl only)
  double kff = 0.0;               // static feedforward (legacy)
  double ff_k0 = 0.0, ff_k1 = 0.0, ff_k2 = 0.0; // static + dynamic FF
  bool spf = false;               // enable setpoint filter
  double spf_wn = 0.0;            // setpoint filter wn
  double spf_zeta = 1.0;          // setpoint filter zeta
  // Reference governor (C impl only)
  bool rg = false;
  double rg_k = 0.0;              // governor gain
  double rg_sens = 1.0;           // sensitivity normalization
  double rg_max_rate = 0.0;       // cap on setpoint rate (abs)
  // Optional clamping and trims
  double i_deadband = 0.0;        // pause I when |e| <= deadband
  double d_clip = 0.0;            // clip |u_d| <= d_clip
  // limits
  double umin = -1.0, umax = 1.0;
  // settle detection hold duration
  double settle_hold = 0.5;       // seconds continuously in-band
  std::string csv;
};

static bool starts_with(const char* s, const char* p) {
  return std::strncmp(s, p, std::strlen(p)) == 0;
}

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const char* s = argv[i];
    auto need = [&](int n){ if (i + n >= argc) { std::cerr << "missing value for " << s << "\n"; std::exit(2);} };
    if (starts_with(s, "--impl=")) a.impl = std::string(s + std::strlen("--impl="));
    else if (starts_with(s, "--plant=")) a.plant = std::string(s + std::strlen("--plant="));
    else if (starts_with(s, "--steps=")) a.steps = std::atoi(s+8);
    else if (starts_with(s, "--dt=")) a.dt = std::atof(s+5);
    else if (starts_with(s, "--sp=")) a.sp = std::atof(s+5);
    else if (starts_with(s, "--tau=")) a.tau = std::atof(s+6);
    else if (starts_with(s, "--K=")) a.K = std::atof(s+4);
    else if (starts_with(s, "--zeta=")) a.zeta = std::atof(s+7);
    else if (starts_with(s, "--wn=")) a.wn = std::atof(s+5);
    else if (starts_with(s, "--kp=")) a.kp = std::atof(s+5);
    else if (starts_with(s, "--ki=")) a.ki = std::atof(s+5);
    else if (starts_with(s, "--kd=")) a.kd = std::atof(s+5);
    else if (starts_with(s, "--beta=")) a.beta = std::atof(s + std::strlen("--beta="));
    else if (starts_with(s, "--gamma=")) a.gamma = std::atof(s + std::strlen("--gamma="));
    else if (starts_with(s, "--kdN=")) a.kdN = std::atof(s + std::strlen("--kdN="));
    else if (starts_with(s, "--kff=")) a.kff = std::atof(s + std::strlen("--kff="));
    else if (starts_with(s, "--ffk0=")) a.ff_k0 = std::atof(s + std::strlen("--ffk0="));
    else if (starts_with(s, "--ffk1=")) a.ff_k1 = std::atof(s + std::strlen("--ffk1="));
    else if (starts_with(s, "--ffk2=")) a.ff_k2 = std::atof(s + std::strlen("--ffk2="));
    else if (std::strcmp(s, "--spf") == 0) a.spf = true;
    else if (starts_with(s, "--spf-wn=")) a.spf_wn = std::atof(s + std::strlen("--spf-wn="));
    else if (starts_with(s, "--spf-zeta=")) a.spf_zeta = std::atof(s + std::strlen("--spf-zeta="));
    else if (std::strcmp(s, "--rg") == 0) a.rg = true;
    else if (starts_with(s, "--rg-k=")) a.rg_k = std::atof(s + std::strlen("--rg-k="));
    else if (starts_with(s, "--rg-sens=")) a.rg_sens = std::atof(s + std::strlen("--rg-sens="));
    else if (starts_with(s, "--rg-max-rate=")) a.rg_max_rate = std::atof(s + std::strlen("--rg-max-rate="));
    else if (starts_with(s, "--i-deadband=")) a.i_deadband = std::atof(s + std::strlen("--i-deadband="));
    else if (starts_with(s, "--d-clip=")) a.d_clip = std::atof(s + std::strlen("--d-clip="));
    else if (starts_with(s, "--umin=")) a.umin = std::atof(s + std::strlen("--umin="));
    else if (starts_with(s, "--umax=")) a.umax = std::atof(s + std::strlen("--umax="));
    else if (starts_with(s, "--settle_hold=")) a.settle_hold = std::atof(s + std::strlen("--settle_hold="));
    else if (starts_with(s, "--csv=")) a.csv = std::string(s + std::strlen("--csv="));
    else if (std::strcmp(s, "-h") == 0 || std::strcmp(s, "--help") == 0) {
      std::cout << "Usage: pid_bench [options]\n"
                   "  --impl=cpp|c         choose PID implementation (default cpp)\n"
                   "  --plant=first|second plant model (default second)\n"
                   "  --steps=N            iterations (default 5000)\n"
                   "  --dt=SEC             timestep (default 0.001)\n"
                   "  --sp=VAL             step setpoint (default 1.0)\n"
                   "  --kp= --ki= --kd=    PID gains\n"
                   "  --tau= --K=          first-order plant params\n"
                   "  --zeta= --wn= --K=   second-order plant params\n"
                   "  --umin= --umax=      saturation (default -1..1)\n"
                   "  --beta= --gamma=     2DOF weights (C impl)\n"
                   "  --kdN=               D lowpass N (C impl)\n"
                   "  --kff=               static feedforward (C impl)\n"
                   "  --ffk0/1/2=         dynamic feedforward gains (C impl)\n"
                   "  --spf [--spf-wn= --spf-zeta=]  setpoint filter (C impl)\n"
                   "  --rg [--rg-k= --rg-sens= --rg-max-rate=]  reference governor (C impl)\n"
                   "  --i-deadband= --d-clip=   trims for I/D (C impl)\n"
                   "  --csv=FILE.csv       dump time,y,u,e to CSV\n";
      std::exit(0);
    }
  }
  return a;
}

struct Metrics {
  double iae = 0;   // integral of absolute error
  double ise = 0;   // integral of squared error
  double rise_t = NAN;      // 10%-90% rise time
  double settle_t = NAN;    // 2% band settle time (with hold)
  double overshoot = 0;     // % overshoot (relative to |sp|)
};

static void update_metrics(
  Metrics& m,
  double t, double dt, double y, double sp,
  double& t10, double& t90,
  double& in_band_time, double settle_hold,
  double& peak_y, double& trough_y,
  double& settle_time_candidate)
{
  const double e = sp - y;
  m.iae += std::abs(e) * dt;
  m.ise += e * e * dt;

  const double amp = std::abs(sp);
  if (amp > 1e-12) {
    // 10-90% rise time thresholds accounting for sign of step
    const double y10 = 0.1 * sp;
    const double y90 = 0.9 * sp;
    if (std::isnan(t10)) {
      if ((sp > 0 && y >= y10) || (sp < 0 && y <= y10)) t10 = t;
    }
    if (std::isnan(t90)) {
      if ((sp > 0 && y >= y90) || (sp < 0 && y <= y90)) t90 = t;
    }

    // Overshoot tracking using peak/trough
    peak_y = std::max(peak_y, y);
    trough_y = std::min(trough_y, y);
    if (sp > 0) {
      m.overshoot = std::max(m.overshoot, std::max(0.0, (peak_y - sp)) / amp * 100.0);
    } else {
      m.overshoot = std::max(m.overshoot, std::max(0.0, (sp - trough_y)) / amp * 100.0);
    }

    // Settle detection with hold time inside ±2% band
    const double band = 0.02 * amp;
    if (std::abs(e) <= band) {
      in_band_time += dt;
      if (std::isnan(m.settle_t) && in_band_time >= settle_hold) {
        // settled since (t - in_band_time)
        settle_time_candidate = t - in_band_time;
        m.settle_t = settle_time_candidate;
      }
    } else {
      in_band_time = 0.0;
    }
  }
}

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv);

  // Controller instances
  PID pid_cpp(a.kp, a.ki, a.kd);
  PID_Controller_t pid_c;
  PID_Init(&pid_c, (float)a.kp, (float)a.ki, (float)a.kd, (float)a.umin, (float)a.umax);
  PID_SetSampleTime(&pid_c, (float)a.dt);
  PID_Set2DOF(&pid_c, (float)a.beta, (float)a.gamma, (float)a.kdN);
  if (a.kff != 0.0) PID_SetFeedforwardGain(&pid_c, (float)a.kff);
  if (a.ff_k0 != 0.0 || a.ff_k1 != 0.0 || a.ff_k2 != 0.0) {
    PID_SetDynamicFeedforward(&pid_c, (float)a.ff_k0, (float)a.ff_k1, (float)a.ff_k2);
  }
  if (a.spf) {
    PID_EnableSetpointFilter(&pid_c, 1);
    if (a.spf_wn > 0.0) {
      PID_ConfigSetpointFilter(&pid_c, (float)a.spf_wn, (float)a.spf_zeta);
    }
  }
  if (a.rg) {
    PID_EnableReferenceGovernor(&pid_c, 1);
    PID_ConfigReferenceGovernor(&pid_c, (float)a.rg_k, (float)a.rg_sens, (float)a.rg_max_rate);
  }
  if (a.i_deadband > 0.0) PID_SetIntegralDeadband(&pid_c, (float)a.i_deadband);
  if (a.d_clip > 0.0) PID_SetDerivativeClip(&pid_c, (float)a.d_clip);

  // Plant states
  double y = 0.0;         // output
  double v = 0.0;         // velocity for second-order

  std::ofstream ofs;
  if (!a.csv.empty()) {
    ofs.open(a.csv);
    ofs << "t,y,u,e\n";
  }

  Metrics m; double t10=NAN, t90=NAN; double in_band_time=0.0; double settle_time_candidate=NAN; double peak_y=-1e300, trough_y=1e300;

  auto t0 = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < a.steps; ++i) {
    const double t = i * a.dt;
    const double sp = a.sp;

    // Controller
    double u = 0.0;
    if (a.impl == "c") {
      u = PID_Update_2DOF(&pid_c, (float)y, (float)sp, (float)a.dt);
    } else {
      u = pid_cpp.compute(sp, y);
    }
    u = std::clamp(u, a.umin, a.umax);

    // Plant update
    if (a.plant == "first") {
      // dy/dt = (-y + K*u)/tau
      const double dy = (-y + a.K * u) / a.tau;
      y += dy * a.dt;
    } else {
      // second-order: x'' + 2*zeta*wn x' + wn^2 x = K*u
      const double acc = a.K * u - (2.0*a.zeta*a.wn)*v - (a.wn*a.wn)*y;
      v += acc * a.dt;
      y += v * a.dt;
    }

    // Metrics
    update_metrics(m, t, a.dt, y, sp, t10, t90, in_band_time, a.settle_hold, peak_y, trough_y, settle_time_candidate);

    if (ofs.is_open()) {
      ofs << t << "," << y << "," << u << "," << (sp - y) << "\n";
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double total_s = std::chrono::duration<double>(t1 - t0).count();
  double ns_per_step = (total_s * 1e9) / std::max(1, a.steps);

  m.rise_t = (std::isnan(t10) || std::isnan(t90)) ? NAN : (t90 - t10);
  // m.settle_t 已在 update_metrics 中设置为首次满足保持时间的时刻

  std::cout.setf(std::ios::fixed); std::cout.precision(6);
  std::cout << "impl=" << a.impl << ", plant=" << a.plant
            << ", steps=" << a.steps << ", dt=" << a.dt << " s\n";
  std::cout << "runtime: " << (total_s*1000.0) << " ms, " << ns_per_step << " ns/step\n";
  std::cout << "IAE=" << m.iae << ", ISE=" << m.ise
            << ", rise_t=" << m.rise_t << " s, settle_t~" << m.settle_t << " s"
            << ", overshoot=" << m.overshoot << " %\n";

  if (ofs.is_open()) {
    std::cout << "csv -> " << a.csv << "\n";
  }

  return 0;
}
