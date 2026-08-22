# v5.0.2 Full-Package Reconciliation

This patch is a publication-surface and reproducibility reconciliation of v5.0.1. It does **not** change the measured GB10 dataset or Revision-2 simulation records.

## Scientific wording corrected

- Narrowed universal-sounding kernel claims to the two tested GB10 software stacks.
- Reframed the ~3.4× figure as a **model-derived effective KV-path / fixed-path byte-cost ratio**, not a direct hardware per-byte measurement.
- Removed unsupported claims that compression universally dominates placement; the supported statement is that **reducing KV traffic is a first-order opportunity**.
- Removed unsupported fixed RoPE-prefetch hit-rate/locality improvements and fixed EMA-policy advantages.
- Removed hard-coded attention-head prevalence percentages from the core chapter; head behavior is now explicitly model/workload dependent.
- Removed universal CXL/direct-GPU-access, fixed-latency, zero-CPU, zero-fetch and achieved-bandwidth claims from live design figures.
- Recast CXL latency/bandwidth values as design assumptions or measurements-to-run rather than v5 results.

## Live figures corrected

- Replaced the old multi-diagram visual appendices embedded in CH00/CH01/CH02 with an evidence-status map.
- Rebuilt CH03 architecture figures around policy/mechanism separation and explicit unmeasured tier boundaries.
- Rebuilt CH04 latency figures to distinguish aggregate bandwidth from transfer-deadline hiding.
- Rebuilt CH05 bandwidth aggregation as an arithmetic ceiling, not achieved throughput.
- Rebuilt CH06 preprocessing, CH08 MoE and CH09 GPU-integration figures as design hypotheses with explicit validation obligations.
- Rebuilt CH10 market figures as a capability landscape; removed first/only/closest/superiority language and stale point-in-time vendor specifications.
- Rebuilt CH11 performance figures around measured/derived/simulated evidence classes.
- Rebuilt CH12 implementation figure to remove retracted hit-rate, TCO and concurrency claims.
- Rebuilt RoPE-prefetch figures to remove unsupported fixed locality/hit-rate numbers.

## Reproducibility fixes

- Fixed simulator/audit script paths so they resolve relative to `scripts/` instead of the caller's working directory.
- Fixed JSON output/input paths to use `data/` consistently.
- `python -m py_compile scripts/*.py` passes.
- `scripts/summarize.py` successfully reads `data/results_v2.json` and reproduces the published Revision-2 tables.
- All JSON files in `data/` parse successfully.
- Checked 650 local HTML references: zero broken local links.

## Packaging fixes

- Updated current figure-source inventory to 58 HTML sources.
- Removed stale “30-diagram visual appendix” labeling after the visual appendices were replaced.
- Version footers reconciled to v5.0.2.
- Distribution ZIP excludes `.git/` metadata; the source archive no longer exposes repository internals unnecessarily.

## Evidence boundary retained

v5.0.2 remains the empirical-correction release. Hierarchical relevance filtering / candidate selection is **not** retroactively claimed as a measured v5 result. That belongs in the next architecture revision.
