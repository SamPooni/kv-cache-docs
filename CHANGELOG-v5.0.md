# Changelog — v5.0 (August 2026)

v5.0 is not a revision of v4.0. It is the first version of this package built on
**measurement**, and the measurement contradicted v4.0's central claims. Those claims are
withdrawn. `RETRACTIONS.md` is the complete record; this file is what changed in the package.

## 1. The package now contains measured evidence

A DGX Spark (GB10 Grace Blackwell, 128 GiB coherent unified LPDDR5X, 273 GB/s spec) was run
on 19 August 2026 with Qwen2.5-7B-Instruct in fp16. Three results carry the new argument:

- **Achieved streaming bandwidth is 236.5 GB/s — 87% of spec.** This retires the ~50%
  "effective bandwidth" haircut earlier drafts applied to theoretical figures. The haircut was
  wrong, and wrong in the pessimistic direction.
- **The decode model is two-term, not one-term.** Fitting step time against resident KV volume
  gives `100.3 + 16.34·KV_GiB` (HuggingFace + SDPA, R² 0.991) and `73.9 + 17.58·KV_GiB`
  (vLLM + FlashAttention-2, R² 0.945). Weights are read at 64–87% of achievable bandwidth,
  **KV at 26–28%**. A KV byte costs **3.4×** a weight byte.
- **The kernel is not the lever.** Two frameworks with different attention implementations and
  different KV layouts land within **8%** of each other on the KV slope, while the weight read
  improves 34% between them.

And the observation that reframes the whole package: **capacity was never the binding
constraint in any configuration measured.** Everything fit in 128 GiB. Throughput still fell
from 150 tok/s at batch 16 × 4K to 46 tok/s at batch 16 × 16K, with memory to spare and
nothing evicted.

Method, both raw data tables, and the two measurement bugs found on the way are in
**Appendix L**. The scripts are in `scripts/`.

## 2. The thesis changed

v4.0 argued that a CXL tier plus an attention-aware eviction policy buys 16× concurrency at a
97% hit rate. v5.0 argues something narrower and better supported:

> **A memory tier does not eliminate the KV memory wall. It changes its shape — from a
> capacity wall into a data-movement wall.**

Four constraints bind independently, and conflating them produces the false conclusion that
adding capacity solves KV scaling: **capacity** (relieved by a tier), **HBM-side bandwidth**
(unchanged by a tier, and measured here), **tier-ingress bandwidth** (created by tiering), and
**per-transfer latency** (created, and distinct from ingress bandwidth — average bandwidth can
be sufficient while individual transfers still stall).

## 3. The contribution changed

Not the scoring function. A decayed-frequency score is **LRFU** (Lee et al., "On the Existence
of a Spectrum of Policies that Subsumes the LRU and LFU Policies", ACM SIGMETRICS 1999), and α
is its documented LRU↔LFU knob — established prior art, not a new algorithm.

What survives is a protocol property:

> **Make KV state and its control metadata a single migratable object, with identity-keyed
> policy state that survives the round trip between tiers.**

In the corrected simulation this is worth **+34.15 pts to LFU** and **+24.85 pts to
EMA α=0.01** at a 32 GiB budget — **larger than the spread between policies**, and it removes
the small-α collapse. It is testable on host DRAM over PCIe before any CXL hardware is bought.
Formal statement in **Appendix M**; results in **Appendix N**.

## 4. A second retraction, inside the simulation

The Revision 1 policy study was itself withdrawn. `_pick_victim()` re-inserted a popped heap
entry only when its recomputed score **exceeded** the stored key — correct for LFU, where
counts rise and keys go stale-low; wrong for a decaying EMA, where scores fall and keys go
stale-high. Audited against brute-force `argmin`: **99.6% wrong victims** for EMA α=0.15.
Four of five findings reversed. Fixed with an exact order-preserving key
`K = ln S_touch + t_touch·(−ln(1−α))`; re-audit gives 0 wrong of 78,860.

Revision 1's simulator is retained unmodified as
`scripts/kv_tiering_sim_rev1_retained_for_audit.py` so the defect stays checkable, and
Revision 2 is labelled **provisional** everywhere it appears — one defect already inverted the
study's conclusions once.

## 5. Structural changes

- **New Appendix L** (hardware measurements), **M** (formal core: object model, control/data
  plane split, objective function), **N** (simulation and erratum). Appendix nav now runs A–N.
- **New `RETRACTIONS.md`** — every withdrawn claim, what it rested on, what contradicted it,
  what replaced it, plus the two measurement bugs and the pattern they share.
- **`CANONICAL-NUMBERS.md` rewritten.** Seven evidence classes, with **`Measured` present for
  the first time**. §7 is the retraction list.
- **All fourteen chapters retitled and rewritten** around the four-constraint argument.
  Chapter 6 became *Quantisation and KV Reduction* — the measurement makes compression the
  first-order lever, since the bytes removed from KV are the expensive ones. Chapter 7 became
  *Block-Granular KV State Management*: the cache object is a **(sequence, token block)**, not
  a head, and the lossless/lossy boundary is now stated formally as `x_b(t) = 1 ∀ b ∈ Read(t)`
  — which is what resolves v4.0's conflation of cache hit rate with attention sparsity.
- **Appendices A–K reconciled**; the retracted derivations in H, I, J and K are replaced by
  pointers to L and N and by explicit statements of what is *not* established.
- **36 figures moved to `figures/superseded-v4.0/`**, each carrying a banner saying it
  illustrates a withdrawn claim. They were retained rather than deleted so the retraction stays
  checkable. Twelve live figures remain, built on measured or clearly-labelled data.
- **`data/` and `scripts/` now ship with the package** — `results_v2.json`, `alpha_grid.json`,
  the measurement scripts and the simulator — so every number is reproducible from the package
  itself.

## 6. Corrections carried in

- **Llama-2-70B is GQA with 8 KV heads** (64 query heads), so 320 KiB/token — the same as
  Llama-3-70B. The 2,560 KiB/token MHA row belongs to Llama-1-65B, not Llama-2-70B.
- The naive **3.66 TB/s** decode requirement and the **11.4 GB/s** steady-state tier demand are
  both gone: the first was a strawman, the second was computed from `1 − 0.97`. No steady-state
  tier demand is stated in v5.0, because there is no measured miss rate to compute one from.

## 7. Verification on this build

81 pages rendered in a fully offline browser: **0 JavaScript errors, 0 failed requests,
0 blank pages**. 0 broken internal links · 0 HTML structural errors · 17/17 Babel blocks
compile · 0 external network requests · **0 WCAG AA contrast failures across 11,727 text
nodes** · `role="main"` on 81/81 pages · 0 iframes without a title.
