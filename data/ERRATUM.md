# Erratum — Revision 1 policy findings withdrawn

Peer review of Revision 1 identified a defect in the eviction data structure and three
experimental-design problems. Correcting them reverses or substantially alters every
policy finding. The KV-memory correction is unaffected.

## The defect

`_pick_victim()` re-inserted a popped heap entry only when its recomputed score
**exceeded** the stored key. Correct for LFU (counts rise; keys stale-low). Wrong for a
decaying EMA (scores fall; keys stale-high) — the popped entry is returned without
checking whether another block decayed further.

Audited against brute-force `argmin` (`bug_check.py`):

| Policy | Wrong victims | Evictions | Rate |
|---|---|---|---|
| EMA α=0.15 | 114,114 | 114,528 | **99.6%** |
| EMA α=0.01 | 73,650 | 74,987 | **98.2%** |
| LFU | 0 | 74,987 | 0.0% |

Fix: exact order-preserving key `K = ln S_touch + t_touch·(−ln(1−α))`, which orders
identically to the current decayed score at any time, so no lazy re-evaluation is needed.
Re-audit (`bug_check2.py`): 0 wrong of 78,860.

## What changed

| Rev 1 finding | Status | Rev 2 |
|---|---|---|
| R1 — EMA α=0.15 worst at every budget | **Reversed** | R1′ — EMA α=0.15 *best* at 4 of 5 budgets (44.18% vs LRU 38.34% @32 GiB) |
| R2 — hit rate monotone as α→0, converges exactly onto LFU | **Withdrawn** | R3′ — non-monotonic, interior optimum α≈0.05 (55.93%); no convergence onto LFU |
| R3 — LRU wins the scan workload | **Withdrawn** | EMA α=0.01 leads scan (36.26% vs 31.51%) |
| R4 — EMA wins the cyclic sweep 2.2× | **Withdrawn** | Only LFU survives loop (22.06%); EMA tracks LRU at 9.83% |
| R5 — 97% of stall is recompute; policy nearly irrelevant | **Withdrawn** | R5′ — warm recompute is 50–67%; policy spread reaches 86% at 128 GiB |
| C1 — 8K Llama-3 70B KV is 2.50 GiB, not 20 GB | **Stands** | Unchanged; analytic, simulator-independent |

## New results

- **R4′ (largest effect in the study).** Letting scoring metadata survive HBM eviction is
  worth **+34.15 pts to LFU** and **+24.85 pts to EMA α=0.01** — more than the spread
  between policies. It also removes the small-α collapse, identifying metadata reset as
  its cause. Best config: EMA α=0.01 persistent @64 GiB = 78.72% vs LRU 67.12%.
- **R2′.** LFU collapses to worst once decode growth is modelled — fresh count-1 blocks
  are evicted in favour of stale high-count blocks (classical LFU aging).
- **R6′.** Under idealised overlap, fetch falls below 1.3 ms/turn for every policy.
  Recompute avoidance is then the only term that matters.

## Standing caveat

A single data-structure defect inverted four of five findings. Revision 2's numbers should
be treated as provisional, not settled. Remaining unaddressed: no Belady bound, no
bandwidth/queueing model, whole-context residency still assumed, synthetic workload,
2–3 seeds, no hardware validation.

`kv_tiering_sim.py` (Revision 1) is retained unmodified in the bundle so the defect and
its effect on the published numbers remain independently checkable.
