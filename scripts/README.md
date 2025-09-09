Scripts layout

- run/: Helpers to run the binaries locally
  - run_x86.sh
  - run_arm64_local.sh
- deploy/: Board-related helpers
  - deploy_to_board.sh
  - sync_sysroot.sh
- tune/: Benchmarking, tuning and plotting
  - auto_tune_pid.py
  - tune_pid.py
  - plot_step_gamma.py

Back-compat wrappers are kept at original locations under `scripts/` so
existing tasks/shortcuts continue to work.
