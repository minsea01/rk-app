# RK3588 Board Runtime Dependencies

This project treats `pyproject.toml` as the dependency source of truth.
The file `requirements_board.txt` is generated from `pyproject.toml` by:

`python3 tools/export_requirements.py`

Dependency files:
- `requirements.txt`: base runtime (lightweight)
- `requirements_board.txt`: base + board extras (`rknn-toolkit-lite2`)
- `requirements_train.txt`: base + training/export extras (`torch`, `ultralytics`, `onnxruntime`, etc.)

## Important: `rknn-toolkit-lite2` Offline Installation

On some RK3588 environments, `rknn-toolkit-lite2` is not available from the default PyPI index.
Install the wheel manually first, then install board requirements:

1. Download the correct wheel from Rockchip releases or your internal mirror.
2. Install it on the board:
   `pip install /path/to/rknn_toolkit2_lite-<version>-py3-none-any.whl`
3. Install base runtime requirements:
   `pip install -r requirements.txt`
4. Install the generated board requirements:
   `pip install -r requirements_board.txt`

If your environment uses a private package index, set `PIP_INDEX_URL` and `PIP_EXTRA_INDEX_URL`
before running `pip install`.
