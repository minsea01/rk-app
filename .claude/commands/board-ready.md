---
description: Check RK3588 board deployment readiness
allowed-tools: "*"
---

Check RK3588 board deployment readiness with comprehensive component verification.

## Task

Perform a complete deployment readiness check:

1. Check ARM64 binary exists and is correct architecture
2. Verify RKNN models exist and meet <5MB requirement
3. Validate configuration files (detect_rknn.yaml)
4. Check deployment scripts are executable
5. List on-device requirements that need hardware verification
6. Generate detailed readiness report

## Expected Actions

- Check binary: `file out/arm64/bin/detect_cli` and `du -h out/arm64/bin/detect_cli`
- Check models: `find artifacts/models -name "*.rknn" -exec du -h {} \;`
- Verify config: `cat config/detection/detect_rknn.yaml`
- Check scripts: `ls -l scripts/deploy/*.sh`
- Generate report: `artifacts/board_ready_report.md`

## Report Should Include

- Component status table (binary, models, config, scripts)
- Graduation requirements compliance checklist
- Deployment instructions for when board arrives
- Risk assessment
- Next steps (immediate and pending hardware)

## Success Criteria

- All checks executed without errors
- Clear status for each component (✅/❌/⏸️)
- Actionable recommendations provided
- Report suitable for thesis documentation
