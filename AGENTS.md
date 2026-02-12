# Repository Guidelines

## Project Structure & Module Organization
- Core C++ pipeline lives in `src/` (capture/preprocess/infer/post/output) with headers under `include/rkapp/`. Example CLI lives in `examples/` (`detect_cli.cpp`).
- Board-facing Python runners are in `apps/` (e.g., `yolov8_rknn_infer.py`, `yolov8_stream.py`), with shared scripts in `tools/` (export/convert/compare) and orchestration helpers in `scripts/` (deploy, bench, validate).
- Models and metrics land in `artifacts/models/` (`<model>[_variant].onnx`, `_int8.rknn`, `_fp16.rknn`) and `artifacts/bench_summary.*`. Datasets/calibration stay in `datasets/` (not committed).
- Runtime and deploy configs are under `config/`; `configs/` mirrors the same layout for compatibility. Keep changes in sync if both paths are used by your workflow. Docker support is in `docker/`. Tests are split across `tests/` (Py + GTest) and CTest targets generated from `tests/*.cpp`.

## Build, Test, and Development Commands
- Host debug build (ONNX): `cmake --preset x86-debug && cmake --build --preset x86-debug`; run CLI: `./build/x86-debug/detect_cli --cfg config/detection/detect.yaml`.
- RK3588 cross build (RKNN): `cmake --preset arm64-release -DENABLE_RKNN=ON && cmake --build --preset arm64`; install: `cmake --install build/arm64`. Prerequisites: `aarch64-linux-gnu-g++`, `toolchain-aarch64.cmake`, and (if needed) `AARCH64_SYSROOT`/`RKNN_HOME`.
- Training → ONNX → RKNN: `make RUN_NAME=<exp> all MODEL_PREFIX=yolo11n`; FP16 only: `make convert-fp16`; compare PC vs RKNN: `make compare COMPARE_IMG=<img>`. Prerequisites: Docker images `yolov8-train:cu121`, `rknn-build:1.7.5`, and `yq`.
- Python runner sample: `python apps/yolov8_rknn_infer.py --model artifacts/models/best.rknn --names config/classes.txt --source assets/test.jpg --save artifacts/vis.jpg`.
- Benchmarks: `scripts/run_bench.sh` (outputs under `artifacts/`).

## Testing Guidelines
- C++: build with the preset above, then `ctest --preset x86-debug` (Sanitizers on by default; use `x86-debug-nosan` if needed). Tests rely on GTest; `FETCH_GTEST` pulls it when missing.
- Python: `pytest tests/unit -m "not requires_hardware" -v`; markers available: `unit`, `integration`, `slow`, `requires_hardware`, `requires_model`. Coverage: `pytest tests/unit --cov=apps --cov=tools --cov-report=term`.
- Place new Python tests under `tests/unit/` as `test_*.py`; C++ tests go in `tests/` or `tests/unit/` and are registered via CMake/CTest.

## Quick Sanity Check
- C++ host check: `cmake --preset x86-debug && cmake --build --preset x86-debug && ctest --preset x86-debug`.
- Python unit check (no hardware): `pytest tests/unit -m "not requires_hardware" -v`.
- RKNN path check: `python apps/yolov8_rknn_infer.py --model artifacts/models/best.rknn --names config/classes.txt --source assets/test.jpg --save artifacts/vis.jpg`.

## Coding Style & Naming Conventions
- C++ follows `.clang-format` (Google style, 2-space indent, 100 cols) and `.clang-tidy` defaults; keep headers under `include/rkapp/` mirroring `src/` layout.
- Python uses 4-space indent and 100-col limit (`pyproject.toml`); run `black`, `isort`, `flake8`, and `bandit` (via `pre-commit run --all-files`).
- Scripts/YAML/JSON use 2-space indent per `.editorconfig`. Keep file/module names snake_case; classes in PascalCase; constants in ALL_CAPS.
- Model artifacts follow `<model>[_variant].onnx`, `<model>[_variant]_int8.rknn`, `<model>[_variant]_fp16.rknn`.

## Commit & Pull Request Guidelines
- Prefer Conventional Commit prefixes (`feat`, `fix`, `docs`, `refactor`, etc.) as seen in git history; keep scope small and descriptive.
- PRs should state the problem, the approach, and test evidence (e.g., `ctest --preset x86-debug`, `pytest -m "not requires_hardware"`, key benchmark outputs). Link issues when applicable and note any model/config changes (include resulting artifact paths).
- Avoid committing large datasets or secrets; `.agent_config.json` and generated artifacts stay local. Update relevant docs (`README.md`, `tests/TESTING_GUIDE.md`) when workflows change.

## Security & Configuration Tips
- Set `ENABLE_ONNX`/`ENABLE_RKNN`/`ENABLE_GIGE` in CMake as needed; point `RKNN_HOME`/`ORT_HOME` to external SDKs if not in defaults.
- Keep API keys and board credentials out of the repo; validate Docker commands (`docker-compose.*`) before use on constrained hosts.
