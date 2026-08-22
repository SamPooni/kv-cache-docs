# Changelog — v6.0

Sam Pooni

---

## Summary

v6.0 reframes this package's central contribution. v5 presented "persist
eviction-policy metadata across tier boundaries" (host owns policy, device
owns mechanism) as the headline result. v6 subordinates that result to a
larger architecture — **The KV-State Control Plane** — and treats the v5
persistence finding as one validated mechanism living inside that
architecture's PLACE stage, not as the whole contribution.

This is a reframe, not a retraction. The v5 numbers that were carried
forward remain unchanged and unretracted (see below). The numbers that were
already retired before v5 stay retired.

## (a) Architectural reframe

**Before (v5):** the headline claim was "identity-keyed eviction-policy
metadata that survives tier eviction, evaluated as LFU at 32 GiB HBM."
Everything else in the package supported that one claim.

**After (v6):** the headline claim is an architecture — The KV-State
Control Plane — built on two orthogonal hierarchies:

- an **information hierarchy** (agent/model state → context-region
  summaries → candidate regions → KV blocks → token-level KV), answering
  "what is worth considering?"
- a **physical hierarchy** (HBM → host/CXL DRAM → NVMe/remote storage),
  answering "where should it live?"

connected by a control plane with three stages — **SELECT, PLACE, MOVE** —
feeding **ATTEND** then **OBSERVE**, with feedback back to agent/model
state.

The v5 result (identity-keyed policy state surviving tier eviction, +34.15
pts simulated/provisional, LFU @ 32 GiB HBM) is now positioned as one
validated mechanism inside **PLACE**: it answers "does previously-computed
residency/policy state survive a tier boundary and improve the next
placement decision," not "what should the whole system do." It is demoted
in scope, not in correctness — the number itself is unchanged and is still
simulated/provisional, exactly as before.

See `ARCHITECTURE-v6.md` for the full architecture reference and Chapter 3
of the main package for its narrative treatment.

## (b) New landscape chapter (Chapter 10)

Chapter 10, "The 2026 Landscape," is new in v6.0. It maps five real,
shipping/published 2026 systems by the specific decision each one
optimizes, and positions this package's control-plane framing as a
description of how those pieces could compose — not as a claim that any of
them are inferior or that a gap exists that this package alone fills.

- **CacheWise** — predicts KV reuse from agent and tool-call signals
  (prefix-aware scheduling + reuse-aware eviction guided by tool-call
  metadata). Reports 2–2.6× fewer KV-cache evictions and up to 3.5× faster
  agent session completion time, per its own vLLM implementation and
  coding-agent traces.
  Source: https://letsdatascience.com/news/cachewise-improves-kvcache-reuse-for-llm-coding-agents-c1a1e786
- **PNM-KV** — selects which KV/token pages are needed before paying the
  movement cost, using a near-memory accelerator. Reports up to 21.9×
  throughput improvement in its own benchmarks (arXiv:2511.00321) — cited
  here as their reported figure, not independently verified by this
  package.
- **LMCache** — multi-tier KV object management: pinning, lookup, cleanup,
  movement, compression, batched movement, compute/I·O pipelining,
  cross-engine transfer, prefill/decode disaggregation.
- **TraCT** — rack-scale KV sharing and transfer over CXL, implemented on
  NVIDIA Dynamo, performing direct GPU↔CXL operations behind a rack-wide,
  prefix-aware cache.
- **Tutti** (arXiv:2605.03375) — efficient GPU↔NVMe KV object movement,
  removing the CPU from the KV data/I·O critical path via GPU io_uring and
  slack-aware I·O scheduling. Against GDS-enabled SSD-backed LMCache,
  reports 78.3% TTFT reduction under strict SLO constraints, 2× higher
  achievable request rate, and 27% lower serving cost.

v6's framing: these five systems specialize different pieces of the same
underlying control-plane problem (relevance, reuse prediction, residency,
movement) rather than competing to solve one uniform problem badly. None of
them are described as inferior or as leaving "a gap" this package
uniquely fills.

## (c) Retracted claims remain retracted

v6.0 does not reintroduce, restate, or rely on any of the following. They
were retired in earlier revisions and stay retired:

- 97% hit rate
- 16× capacity (expansion is now stated as 6×, Analytical, per
  `CANONICAL-NUMBERS.md`)
- 36% CapEx reduction
- 65× latency improvement
- "first complete CXL-native solution" (false on its face once TraCT —
  already a CXL-native, rack-scale, Dynamo-based system — is accounted for)
- "all existing work treats KV cache uniformly" (does not survive contact
  with CacheWise, PNM-KV, LMCache, TraCT, or Tutti, each of which solves a
  distinct, non-trivial sub-problem)

Nothing in this changelog or in v6.0's chapters restores any of the above,
under any name.

## (d) v6 is a proposed architecture, not a validated system

The KV-State Control Plane is a **proposed architecture** — a design
artifact assembled by generalizing roles that CacheWise, PNM-KV, LMCache,
TraCT, and Tutti already play separately, plus this package's own
persistence result. It has not been implemented or measured as a complete
system.

Stated plainly, in the author's own framing: **architecture is a design
artifact; measurement validates performance claims, not whether something
qualifies as an architecture.** Describing SELECT/PLACE/MOVE/ATTEND/OBSERVE
as an architecture is a claim about structure — that these are the right
joints to cut the problem at. It is not a claim that the structure has been
built, benchmarked, or shown to outperform the specialized systems in
Chapter 10 doing pieces of it today. Only one box in the pipeline (PLACE →
residency persistence) carries any empirical evidence at all, and that
evidence is simulated and provisional, not measured on real hardware.

## (e) Evidence-backed vs. open design work

**Evidence-backed:**

| Item | Evidence class |
|---|---|
| DGX Spark GB10 decode-bandwidth decomposition (236.5 GB/s achieved, ~3.4× KV-vs-weight byte cost) | Measured |
| Residency-state persistence across a tier boundary (+34.15 pts, LFU @ 32 GiB HBM) | Simulated — provisional |

**Open design work (not yet evidence-backed):**

- SELECT via relevance and predictive reuse signals (see the past-only vs.
  predictive signal taxonomy in `ARCHITECTURE-v6.md`)
- MOVE cost and hiding/overlap behavior on real CXL hardware
- Representation choices for context-region summaries and candidate
  regions in the information hierarchy
- Mechanism placement — which stage (SELECT, PLACE, or MOVE) specific
  techniques from Chapter 10's landscape (e.g., tool-call-driven reuse
  prediction, near-memory selection) should occupy in a unified
  implementation
- The feedback loop from OBSERVE back to model/agent state

None of this open work is claimed as validated in v6.0. See
`ARCHITECTURE-v6.md` for the complete evidence table.

---

© 2025–2026 Subramaniyam (Sam) Pooni. All Rights Reserved.
