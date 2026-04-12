# UI Review

Date: 2026-04-10
Scope: `presentation/web/*`

## 6-Pillar Audit

### 1. Clarity
Score: 4/4

- Product title now matches mission: `Hệ thống trinh sát âm thanh`.
- Technical internals such as ProtonX are hidden behind user-facing language.
- Workflow is split into input, transcript refinement, and intelligence analysis.

### 2. Information Hierarchy
Score: 4/4

- Primary hero explains purpose, operating mode, and hardware expectation.
- Form is grouped by task intent instead of raw technical switches.
- Results panel separates raw transcript, stabilized transcript, corrected transcript, and intelligence brief.

### 3. UX Flow
Score: 4/4

- Step buttons follow the recommended order of operations.
- Intelligence summary is a distinct optional step, not mixed into transcription.
- Scenario selection is surfaced before execution so later analysis uses a saved job context.

### 4. Language & Domain Fit
Score: 4/4

- Labels now speak the language of điều tra / trinh sát rather than internal component names.
- Threat level, classification, and investigator note are promoted in the UI.
- Summary output is rendered as an operational brief instead of raw JSON.

### 5. Visual System
Score: 3/4

- Introduced a consistent palette, card hierarchy, typography contrast, and result badges.
- Improved spacing, panel rhythm, button hierarchy, and mobile responsiveness.
- Remaining gap: a future iteration could add timeline visualization for speaker/intel correlation.

### 6. Offline Trust
Score: 4/4

- UI now states local-only behavior explicitly.
- New Stable-TS and intelligence summary flows rely on local assets and local models only.
- CPU-safe wording remains visible to reduce accidental interference with other workloads.

## Main Fixes Applied

- Renamed UI title and product framing.
- Replaced ProtonX label with `Giảm ảo giác ngữ cảnh`.
- Added `Tóm tắt trinh sát` workflow and result panel.
- Added scenario selector for intelligence analysis.
- Improved model readiness cards and stage labels to use human-readable domain language.

## Next Candidate Iteration

- Add speaker timeline + evidence markers.
- Add domain-specific quick presets for common investigative workflows.
