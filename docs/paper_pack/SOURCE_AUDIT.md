# Source Audit

Date: 2026-04-11

## Purpose

This audit prevents the paper from inheriting mixed-quality claims from older research notes.

Status labels:

- `Keep`: safe supporting source for the current paper pack
- `Supersede`: contains useful ideas but should not be cited directly for the current paper narrative
- `Reference only`: background material, not authoritative for paper claims

## Current Docs Review

| Document | Status | Reason |
| --- | --- | --- |
| `docs/PHOWHISPER_CPP_RESEARCH_2026-04-11.md` | Keep | recent, aligned with the systems-contribution framing |
| `docs/PHOWHISPER_CPP_EXPERIMENT_PROTOCOL.md` | Keep | strong protocol note for the runtime study |
| `docs/PHOWHISPER_CPP_COLAB_PRO_WORKFLOW.md` | Keep | useful reproducibility runbook |
| `docs/OSS_STT_ASR_INTEGRATION_REVIEW_2026-04-10.md` | Keep | focused integration review with practical repository relevance |
| `docs/SPEECHTOINFOMATION_UPSTREAM_REVIEW_2026-04-11.md` | Reference only | useful comparison note, but not a core paper source |
| `docs/WHISPER_MODEL_SELECTION.md` | Supersede | mixed language, older framing, and too proposal-like for the final pack |
| `docs/ADVANCED_RESEARCH_PROPOSAL.md` | Supersede | broad proposal document with mixed certainty levels |
| `docs/DEEP_RESEARCH_UPGRADE_2026.md` | Supersede | useful ideas, but not strict enough for paper claims |
| `docs/SOTA_UPGRADE_PROPOSAL.md` | Supersede | roadmap style rather than evidence-driven paper documentation |
| `docs/UPGRADE_RESEARCH_2026.md` | Supersede | broad research note with mixed claim strength |
| `docs/TECHNICAL_RECOMMENDATIONS.md` | Reference only | engineering guidance, not paper authority |
| `docs/EVALUATION_REPORT.md` | Reference only | may contain useful details but needs cross-check before reuse |
| `docs/Tai_lieu_tong_hop_ky_thuat_Cherry_Core_2026.pdf` | Reference only | summary material, not canonical source |
| `docs/Tai_lieu_tong_hop_ky_thuat_Cherry_Core_2026.docx` | Reference only | summary material, not canonical source |

## External and Legacy Assets

| Source | Status | Reason |
| --- | --- | --- |
| `E:/Freelance/Research/D12_02.2026_NCKH2026/ASR/artifacts/benchmarks/whispercpp/...` | Reference only | valuable pilot evidence but still external and caveated |
| `E:/Freelance/Research/D12_02.2026_NCKH2026/ASR/notebooks/colab/...` | Reference only | useful for rerun design, not final evidence by itself |
| community `ct2` or `ggml` PhoWhisper releases | Reference only | helpful for feasibility, not authoritative for paper results |

## Operating Rule

When writing the paper:

1. Start from the files in `docs/paper_pack/`.
2. Reuse older docs only when the point is also registered in `CLAIM_MATRIX.md`.
3. Reuse older benchmark numbers only when the source path is copied into `EVIDENCE_REGISTRY.md` with the correct caveat label.
