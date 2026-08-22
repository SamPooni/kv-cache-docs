# CANONICAL NUMBERS — v5.0 (August 2026)

Single source of truth. Every chapter, appendix and figure must use these values verbatim.

**What changed from v4.0.** v4.0 contained no measurements. Its headline set
(16× users, 97% hit rate, 65×, 11.5×, 15.6×) was an internally consistent analytical
construction that hardware measurement and a corrected simulation do not support. Those
figures are **retracted** — see §7 and `RETRACTIONS.md`. v5.0 is built from measured data
where measured data exists.

**Evidence classes, strongest first.** For the first time the package contains `Measured`.

| Class | Meaning | Where |
|---|---|---|
| `Measured` | Run on named hardware, method and raw data published | §1 — DGX Spark GB10 |
| `Simulated (provisional)` | Trace-driven simulation, corrected after an erratum, not settled | §3 — KV tiering sim Rev 2 |
| `Analytical model` | Derived from stated inputs | §2, §4 |
| `Vendor specification` | Copied from a product datasheet | link and device specs |
| `External literature` | A third party's published result, cited not reproduced | related work |
| `Design target` | Desired, not demonstrated | §5 |
| `Illustrative` | Pedagogical, never a result | worked examples |

---

## 1. MEASURED — DGX Spark (GB10), 19 August 2026

Platform: DGX Spark, GB10 Grace Blackwell, 128 GiB coherent unified LPDDR5X, 273 GB/s spec.
Model under test: **Qwen2.5-7B-Instruct, fp16** — 28 layers, 4 KV heads, head_dim 128
→ **56 KiB/token**; weights 15.2 GB = 14.16 GiB.
Scripts: `bw_wall.py`, `vllm_slope.py`. Full method and raw tables: Appendix L.

### 1.1 Achieved streaming bandwidth
```
236.5 GB/s          = 87% of the 273 GB/s spec figure
```
Streaming copy (read + write), 4 GiB buffers, median of 20.
**This retires the ~50% "effective bandwidth" haircut used in earlier drafts.** The haircut
was wrong, and wrong in the pessimistic direction.

### 1.2 The decode model is two-term, not one-term
`t = a + b·KV_GiB`, fitted over a batch × context sweep:

| Framework | Fit (ms) | n | R² | Inferred fixed-path BW | Inferred KV BW |
|---|---|---|---|---|---|
| HuggingFace + SDPA, unpaged | `100.3 + 16.34·KV_GiB` | 7 | **0.991** | 151.5 GB/s | **65.7 GB/s** |
| vLLM 0.20.1 + FlashAttention-2, paged | `73.9 + 17.58·KV_GiB` | 9 | 0.945 | 205.7 GB/s | **61.1 GB/s** |

As a fraction of the 236.5 GB/s actually achieved:
`weights 64% (HF) / 87% (vLLM)` · `KV 28% (HF) / 26% (vLLM)`

### 1.3 The results that carry the argument
- **KV is read at roughly a quarter of achievable memory bandwidth.**
- **Under the two-term fit, the inferred effective KV path is ~3.4× more costly per byte than the fixed path** on this hardware (205.7 ÷ 61.1). This is a model-derived ratio, not a direct byte-level hardware measurement.
- **Kernel substitution alone did not eliminate the measured KV slope.** Two tested frameworks with different attention implementations and paged vs unpaged layout produced KV slopes within **8%** (16.34 vs 17.58 ms/GiB), while the inferred fixed-path bandwidth improved **34%** between them. This does not generalize to all kernels, GPUs, models, or decode regimes.
- **Capacity was not the binding constraint in any configuration measured.** Everything fit
  in 128 GiB; throughput still fell from **150 tok/s at batch 16 × 4K to 46 tok/s at
  batch 16 × 16K**, with memory to spare and nothing evicted.

### 1.4 Provenance discipline — two bugs found and fixed
1. `torch_dtype=` is deprecated and was silently ignored → fp32 load (28.4 GiB) → OOM.
   Unified memory surfaces a CUDA OOM as a Linux OOM kill, hiding the cause.
2. Identical batch token ids let vLLM's prefix cache store **one shared copy** of the KV,
   inflating the x-axis by a factor of B and yielding an apparent KV bandwidth of
   **342.9 GB/s** — above the 273 GB/s bus, which is what exposed it.
   **Any measured bandwidth above the bus is a bug, always.**
   The instructive part: it produced a *better* number supporting a more exciting
   conclusion. It was caught by a physical bound, not by inspection.

### 1.5 Limits on the measurement
One model, one size, one machine. Fixed-length uniform batches, no continuous batching.
vLLM fit noisier (R² 0.945); the 16×16384 point sits 26 ms above the fit and needs a repeat.
No thermal log captured. FlashAttention-2, not 3. **fp8 KV not yet measured** — that run is
the direct test of the compression claim and is the next experiment.

---

## 2. ANALYTICAL — capacity sizing

```
bytes_per_token = 2 × layers × kv_heads × head_dim × bytes_per_element
```
The decisive term is **kv_heads**, not query heads.

| Model | Attention | KV heads | KiB/token | 8K | 128K |
|---|---|---|---|---|---|
| Llama-1 65B (fp16) | MHA | 64 | 2,560 | 20.0 GiB | 320 GiB |
| **Llama-2 70B** (fp16) | **GQA** | **8** | **320** | **2.50 GiB** | **40.0 GiB** |
| **Llama-3 70B** (fp16) | GQA | 8 | 320 | **2.50 GiB** | **40.0 GiB** |
| Llama-3 70B (fp8) | GQA | 8 | 160 | 1.25 GiB | 20.0 GiB |
| Llama-3 8B (fp16) | GQA | 8 | 128 | 1.00 GiB | 16.0 GiB |
| Qwen2.5-7B (fp16) | GQA | 4 | 56 | 0.44 GiB | 7.0 GiB |

**Correction carried into rev5:** Llama-2-70B is **GQA with 8 KV heads** (64 query heads),
not MHA. The 2,560 KiB/token row belongs to Llama-1-65B and the smaller MHA Llama-2
variants, not to Llama-2-70B. Both Llama-2-70B and Llama-3-70B are 320 KiB/token.

**When does KV exceed the weights?** Llama-3 70B bf16 = 140 GB = 130.4 GiB.
```
 32K →  10.0 GiB KV  ( 8% of weights)
128K →  40.0 GiB KV  (31% of weights)
single-sequence crossover ≈ 427,000 tokens
```

Block granularity: **16 tokens = 5.00 MiB** at 320 KiB/token.

---

## 3. SIMULATED (PROVISIONAL) — KV tiering, Revision 2

Llama-3 70B GQA, 16-token blocks, 200 sessions, Zipf(1.1) return frequency, 1–12K contexts,
shared 2048-token prefix, 2500 turns, **421 GiB total footprint**, HBM tier over a 512 GiB
CXL tier. 2–3 seeds. Scripts `kv_tiering_sim_v2.py`, data `results_v2.json`.

> **Standing caveat, required wherever these numbers appear.** A single data-structure
> defect in Revision 1 inverted four of five findings (99.6% wrong victims for EMA α=0.15).
> Revision 2 is **provisional, not settled**: no Belady bound, no bandwidth/queueing model,
> whole-context residency assumed, synthetic workload, 2–3 seeds, no hardware validation.

### 3.1 Hit rate vs HBM budget (no persistent metadata, no overlap)

| HBM | LRU | LFU | EMA α=0.15 | EMA α=0.01 |
|---|---|---|---|---|
| 8 GiB | 8.52 | 8.76 | **12.35** | 12.12 |
| 16 GiB | 17.87 | 12.70 | **22.58** | 18.70 |
| 32 GiB | 38.34 | 16.09 | **44.18** | 35.91 |
| 64 GiB | 67.12 | 23.00 | **69.65** | 66.83 |
| 128 GiB | 88.01 | 32.70 | 88.14 | **91.59** |

EMA α=0.15 leads at 4 of 5 budgets; EMA α=0.01 leads at 128 GiB. **The margin over LRU is
+2.5 to +5.8 points, not +25.** LFU collapses once decode growth is modelled — classical
LFU aging, fresh count-1 blocks evicted in favour of stale high-count blocks.

### 3.2 α is non-monotonic with an interior optimum (32 GiB)

| α | 0.5 | 0.3 | 0.15 | **0.05** | 0.01 | 0.003 | 0.001 | LFU | LRU |
|---|---|---|---|---|---|---|---|---|---|
| hit % | 37.20 | 38.84 | 44.33 | **55.96** | 31.54 | 25.47 | 24.44 | 15.56 | 36.85 |

**α ≈ 0.05 is the optimum, +19.1 points over LRU.** There is no convergence onto LFU as
α → 0; the Revision 1 claim that there was is withdrawn.

> **Why §3.1 and §3.2 differ at 32 GiB** (LRU 38.34 vs 36.85, EMA α=0.15 44.18 vs 44.33,
> LFU 16.09 vs 15.56): they are different experiments in `results_v2.json` — §3.1 is `e1`
> (2 seeds), §3.2 is `e2` (1 seed, the α grid). Quote each table against its own baseline;
> never mix a row from one with a row from the other. The spread between them is itself a
> useful indication of seed noise at this scale.
>
> **Where this spec and `ERRATUM.md` differ**, the spec is authoritative: it is computed
> directly from `results_v2.json`. The erratum quotes 55.93% for the α optimum (spec: 55.96%)
> and 50–67% for warm recompute (spec: 44–89% across all budgets and policies). The erratum
> was written from a narrower slice.

### 3.3 The largest effect in the study — persistent scoring metadata (32 GiB)

Letting identity-keyed policy state survive HBM eviction, rather than re-admitting a block
as if newly created:

| Policy | Metadata reset on eviction | Metadata persists | Δ |
|---|---|---|---|
| LFU | 15.56% | 49.71% | **+34.15 pts** |
| EMA α=0.01 | 31.54% | 56.39% | **+24.85 pts** |
| EMA α=0.15 | 44.33% | 44.12% | −0.21 pts |

**Persistence is worth more than the choice of policy.** It also removes the small-α
collapse, identifying metadata reset as its cause. This is the `formal_core` §1.3 claim,
and it is a **protocol** result — testable with no CXL hardware.

### 3.4 Workload dependence (32 GiB, 3 seeds)

| Workload | LRU | LFU | EMA α=0.15 | EMA α=0.01 |
|---|---|---|---|---|
| zipf (returning sessions) | 39.08 | 15.12 | **44.44** | 38.87 |
| scan (one-shot flood) | 31.51 | 12.52 | 32.57 | **36.26** |
| loop (cyclic sweep) | 9.83 | **22.06** | 9.83 | 9.83 |

Only LFU survives the cyclic sweep. **No policy dominates across all three.**

### 3.5 Where turn stall goes
Recompute is **44–89%** of stall depending on budget and policy — not the 97% claimed in
Revision 1. Under idealised transfer/compute overlap, fetch falls to **0.35–1.25 ms/turn**
and recompute becomes **>99%** of stall: once movement is hidden, recompute avoidance is
the only term that matters.

---

## 4. ANALYTICAL — tier economics

Per 5.00 MiB block (16 tokens, Llama-3 70B GQA):

| Operation | Cost | Basis |
|---|---|---|
| Tier fetch @ 121 GB/s | 43 µs | CXL 3.0 ×16 theoretical, one direction |
| Tier fetch @ **64 GB/s** | **82 µs** | mid-range assumption — **no CXL hardware was measured** |
| Tier fetch @ 32 GB/s | 164 µs | pessimistic |
| Forward compute, 16 tokens, linear term | 9 / 11 / 18 ms | 50% / 40% / 25% MFU |
| Regeneration of an arbitrary *interior* block | ≥ the above, potentially far higher | interior KV is not independently reproducible — hidden states depend recursively on all preceding context |

**Moving a block is two to three orders of magnitude cheaper than regenerating it.** Claim
the order of magnitude, not a precise multiple — the two quantities do not measure the same
operation. The 218× figure is one point in that range (17.9 ms ÷ 82 µs) and must always be
quoted with its inputs.

---

## 5. The four constraints — the organising claim

| Wall | Statement | Effect of adding a memory tier |
|---|---|---|
| **Capacity** | HBM cannot hold the KV of all sequences you want resident | **Relieved** — this is what the tier is for |
| **Bandwidth (a) — HBM-side** | Attention re-reads active KV out of HBM every decode step | **Observed on GB10** — measured in §1.2; the two tested stacks did not remove it |
| **Bandwidth (b) — tier-ingress** | KV must cross the tier boundary fast enough to feed attention | **Created** |
| **Transfer / latency** | Each transfer must complete before compute stalls on it | **Created**, and distinct from (b) |

> **A memory tier does not eliminate the KV memory wall. It changes its shape — from a
> capacity wall into a data-movement wall.**

Sustainability (aggregate rate across the decode set) and hiding (per-transfer) are
independent: the first can hold while the second fails, producing stalls at satisfactory
average bandwidth.

---

## 6. The contribution, stated at the right level

Not the scoring function — a decayed-frequency score is **LRFU** (Lee et al., SIGMETRICS
1999) and α is its documented LRU↔LFU knob. The claim is:

> **Make KV state and its control metadata a single migratable object, with identity-keyed
> policy state that survives the round trip between tiers.**

Host owns policy; device owns mechanism. Payload overhead is negligible (64 B against a
5.00 MiB block = 0.0012%); **metadata processing overhead is not established and must be
measured**. Whether the controller must be *on the device* is open, with four falsifiable
claims attached — and only the fourth (policy state surviving eviction) needs device
residency at all. If the other three fail, the contribution is the protocol and CXL is
merely where it runs.

---

## 7. RETRACTED — do not use anywhere

| v4.0 figure | Status |
|---|---|
| 97% HBM hit rate | **Retracted.** Never measured. Corrected simulation gives 8.5–91.6% depending on budget, policy and workload. |
| The 72 → 80 → 87 → 93 → 97 ladder, +8/+7/+6/+4 | **Retracted.** No ablation ever produced it; the increments were assumed additive and independent. |
| +25 points over LRU | **Retracted.** Measured spread is +2.5 to +5.8 pts at the specified α, +19.1 pts at the tuned α ≈ 0.05 optimum. |
| 16× user capacity | **Retracted.** Downstream of the 97%. Under dense attention every block of an active sequence is read every step. |
| 5.4% hot set → 97% of accesses | **Retracted.** Conflated cache hit rate with attention sparsity — two different claims with different obligations. |
| 65× (200 ns vs 13.0 µs) | **Retracted as a headline.** Compares a hypothetical load/store path against a driver-mediated DMA path. No CXL hardware was measured. |
| 11.5× effective end-to-end latency | **Retracted.** Derived from the 97%. |
| 15.6× TTFT | **Retracted.** Rested on a 16.4 GB/s baseline assumption that the 87%-of-spec measurement contradicts. |
| 36% CapEx reduction | **Retracted.** Priced a configuration justified by the 16×. |
| 200 ns CXL access | **Demoted** to an unmeasured assumption inside a 32–121 GB/s range. |
| 6× memory expansion | **Stands** — arithmetic over stated capacities; never depended on the hit rate. |
| 320 KiB/token, 5.00 MiB blocks, the capacity table | **Stands** — analytic, simulator-independent, independently confirmed. |

Nothing retracted here was dishonest; all of it was constructed rather than observed, and
the construction did not survive contact with a measurement. That is the correct outcome,
and the package should say so plainly.
