# Board Ready - 板端部署就绪状态检查

Comprehensive checklist to verify RK3588 board deployment readiness.

## What this skill does

1. **Check ARM64 Binary**: Verify detect_cli is built and executable
2. **Check RKNN Models**: Verify .rknn models exist and meet size requirements
3. **Check Configuration**: Validate YAML config files
4. **Check Scripts**: Verify deployment scripts are executable
5. **Check Dependencies**: List required libraries and their availability
6. **Generate Deployment Checklist**: Create a markdown report with all findings

## Parameters

- `verbose` (optional): Show detailed information for each check

## Expected Output

- `artifacts/board_ready_report.md` - Deployment readiness report with:
  - ✅/❌ Binary status
  - ✅/❌ Model files status
  - ✅/❌ Configuration status
  - ✅/❌ Script status
  - Deployment instructions
  - Missing items checklist

## Usage

Invoke this skill when:
- Before attempting board deployment
- After building ARM64 binaries
- To verify graduation thesis technical requirements
- To prepare deployment documentation

## Checks Performed

### 1. ARM64 Binary Check
- Location: `out/arm64/bin/detect_cli`
- Executable permissions
- File size and architecture verification

### 2. RKNN Model Check
- Models in `artifacts/models/*.rknn`
- Model size <5MB requirement
- Model file integrity

### 3. Configuration Check
- `config/detection/detect_rknn.yaml` exists and valid
- Source path points to existing assets
- Model path is correct
- Classes file exists

### 4. Deployment Script Check
- `scripts/deploy/rk3588_run.sh` is executable
- `scripts/deploy/deploy_to_board.sh` is executable
- Script syntax validation

### 5. Library Dependencies
- Required libraries list:
  - librknn_api.so (RKNN runtime)
  - libopencv*.so (OpenCV)
  - libyaml-cpp.so (YAML parser)
- Expected locations and RPATH settings

### 6. On-Device Requirements
- Python 3.x with rknnlite
- RKNN_HOME environment variable
- Network configuration (dual-NIC)

## Success Criteria

- ✅ All critical checks pass
- ✅ Deployment script ready to execute
- ✅ Clear instructions for on-device setup
- ✅ Missing items documented with resolution steps

## Output Example

```markdown
# RK3588 Board Deployment Readiness Report

## Summary
- Overall Status: ⚠️ Ready for deployment (minor issues)
- Timestamp: 2025-10-28

## Detailed Checks

### ✅ ARM64 Binary
- detect_cli exists at out/arm64/bin/detect_cli
- Size: 2.3MB
- Architecture: aarch64

### ✅ RKNN Models
- best.rknn: 4.7MB (meets <5MB requirement)
- yolo11n_int8.rknn: 4.3MB

### ⚠️ Configuration
- detect_rknn.yaml: Valid
- Warning: Source path uses relative path (OK)

### ✅ Deployment Scripts
- rk3588_run.sh: Executable, syntax valid
- deploy_to_board.sh: Executable

### ⏸️ Board Requirements (Cannot verify without hardware)
- RKNN runtime library
- OpenCV libraries
- Network configuration

## Deployment Instructions
1. Copy binaries: scp -r out/arm64/* board:/opt/rkapp/
2. Copy models: scp artifacts/models/*.rknn board:/opt/rkapp/models/
3. On board: ./rk3588_run.sh
```
