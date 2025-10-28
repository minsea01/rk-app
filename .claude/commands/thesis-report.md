Generate comprehensive graduation thesis progress report aligned with project requirements.

## Task

Create a detailed progress report for the graduation design project:

1. Analyze current project status (completed vs pending)
2. Calculate completion percentage based on graduation requirements
3. Summarize technical metrics (model size, performance, test coverage)
4. Compare actual progress vs timeline milestones
5. Identify risks and provide recommendations
6. Generate bilingual (Chinese/English) report

## Report Type

Ask user which report type (default: "progress"):
- "progress" - General progress update
- "midterm1" - First midterm check (Nov-Dec 2025)
- "midterm2" - Second midterm check (Jan-Apr 2026)
- "final" - Final thesis documentation

## Expected Content

### Executive Summary
- Project title and current phase
- Completion percentage
- Key achievements
- Critical issues

### Technical Requirements Table
| Requirement | Target | Current | Status |
|-------------|--------|---------|--------|
| System Migration | Ubuntu 20.04/22.04 | WSL2 22.04 | ✅ |
| Model Size | <5MB | 4.7MB | ✅ |
| FPS | >30 | TBD | ⏸️ |
| mAP@0.5 | >90% | TBD | ⏸️ |
| Dual-NIC | ≥900Mbps | Not started | ❌ |

### Completion by Phase
- Phase 1 (Literature): Status
- Phase 2 (System Migration): Status
- Phase 3 (Model Deployment): Status
- Phase 4 (Dataset & Documentation): Status

### Performance Metrics
- PC validation results
- Expected board performance
- Comparison tables

### Risk Assessment
- Hardware availability
- Timeline risks
- Mitigation strategies

### Next Steps
- Immediate actions
- Parallel work
- Hardware-dependent tasks

## Output

Save to: `docs/thesis_progress_report_{date}.md`

Format suitable for:
- Midterm progress reports (中期检查报告)
- Advisor meetings
- Thesis appendix
- Defense preparation
