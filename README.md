# KV-State Management Architecture

**The KV-State Control Plane — A Proposed Architecture for KV-Cache Relevance, Residency, and Movement in LLM Inference**

Sam Pooni

---

## Overview

This package documents a proposed architecture — **The KV-State Control
Plane** — for deciding what KV state is relevant, where it should reside,
and when it should move, grounded in hardware-measured decode-bandwidth
costs (DGX Spark GB10) and a provisional simulation of one mechanism within
it (residency-state persistence). The architecture is built on two
orthogonal hierarchies — an information hierarchy (what is worth
considering) and a physical hierarchy (where it should live) — connected by
a control plane of three stages (SELECT, PLACE, MOVE) feeding ATTEND and
OBSERVE, with feedback to agent/model state. The architecture itself is a
design artifact and has not been validated as a complete system; only the
DGX Spark bandwidth measurements and the residency-persistence simulation
carry empirical evidence, and that evidence is explicitly evidence-tagged
throughout the package. The documentation is organized as 14 chapters
(ch00–ch13) plus a set of supporting appendices, figures, scripts, and data
files.

See `ARCHITECTURE-v6.md` for the formal architecture reference and
`CHANGELOG-v6.0.md` for what changed from v5's persistence-focused framing
to v6's control-plane architecture.

## Key Results

| Metric | Value | Evidence class |
|--------|-------|-----------------|
| Achieved streaming bandwidth | 236.5 GB/s — 87% of 273 GB/s spec | Measured — DGX Spark GB10 |
| KV byte cost vs weight byte | ~3.4× (model-derived from two-term decode fit) | Measured |
| Residency-state persistence gain (LFU, 32 GiB HBM) | +34.15 pts | Simulated — provisional |
| Memory expansion | 6× | Analytical |
| KV size, Llama-3 70B GQA | 320 KiB/token; 5.00 MiB per 16-token block | Analytical |

See `CANONICAL-NUMBERS.md` for the complete, authoritative set of every number in this
package with its evidence class.

## Chapter Contents

| Chapter | Title |
|---------|-------|
| ch00 | Executive Summary |
| ch01 | Introduction |
| ch02 | Background |
| ch03 | The KV-State Control Plane |
| ch04 | Hardware Measurement Methodology and Results |
| ch05 | Bandwidth and Tier Economics |
| ch06 | Preprocessing and Prefill Considerations |
| ch07 | KV State Management — The Protocol Argument |
| ch08 | Mixture-of-Experts and KV Cache Interaction |
| ch09 | GPU and Controller Integration |
| ch10 | The 2026 Landscape |
| ch11 | Results Summary |
| ch12 | A Decision Procedure |
| ch13 | Conclusion |

Appendices A–N provide supporting technical deep-dives referenced from the chapters
above.

## File Structure

```
├── index.html          # Main landing page
├── ARCHITECTURE-v6.md   # Formal architecture reference (companion to Chapter 3)
├── CHANGELOG-v6.0.md    # v5 → v6 transition record
├── css/style.css        # Shared stylesheet
├── chapters/            # ch00-ch13 chapter HTML files
├── appendix/             # Appendix HTML files (A-N)
├── figures/              # Supporting figure sources, organized by chapter
├── scripts/              # Measurement and simulation code
└── data/                 # Raw and processed results
```

## Reproducibility

The raw measurement and simulation artifacts behind the numbers in this package live
in `scripts/` and `data/`.

`scripts/` includes the bandwidth-wall measurement and simulation drivers, among them
`bw_wall.py`, `vllm_slope.py`, and `kv_tiering_sim_v2.py`, along with supporting sweep,
grid-search, and bug-check utilities (`alpha_grid.py`, `cxl_sweep.py`, `final_sweep.py`,
`sweep.py`, `suite_par.py`, `summarize.py`, `adversarial.py`, `bug_check.py`,
`bug_check2.py`). A prior simulator revision is retained for audit as
`kv_tiering_sim_rev1_retained_for_audit.py`.

`data/` holds the corresponding outputs and source records, including
`results_v2.json`, `alpha_grid.json`, `cxl_sweep.json`, `measured_results.md`,
`chapter_section_rev5.md`, `formal_core.md`, and `ERRATUM.md`. A superseded sweep is
retained for audit as `sweep_rev1_superseded.json`.

See `CANONICAL-NUMBERS.md` for the complete, authoritative set of every number in this
package with its evidence class.

---

© 2025–2026 Subramaniyam (Sam) Pooni. All Rights Reserved.
