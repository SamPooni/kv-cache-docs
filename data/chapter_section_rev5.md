# KV State Management Across a Heterogeneous Memory Hierarchy

Replacement draft for *LLM Symphony*, Movements IV & V. **Revision 5 — first revision containing hardware measurements.**

**What changed and why.** Revision 5 replaces the two central assumptions with measurements
taken on a DGX Spark (GB10, 128 GiB unified LPDDR5X). Achieved streaming bandwidth is
**236.5 GB/s — 87% of spec**, not the ~50% haircut previously assumed. And the single-bandwidth
decode roofline is **refuted**: KV is read at roughly a quarter of achievable bandwidth while
weights reach 87%, a gap that persists across two frameworks and two attention kernels. Full
data and method in the companion `measured_results.md`. Revision 4's corrections were: an over-claimed "whole lossless design space," a mathematically wrong movement-
amplification floor, an over-absolute "admission not eviction," "currently scheduled sequence"
where continuous batching means *decode set*, streaming constraints stated for one sequence
instead of the concurrent set, an unsupported 1.5 threshold, and a "should not be argued about"
that no systems reviewer should be told. The formal model — object, plane split, objective
function — is specified in the companion document `formal_core.md` and referenced from §7.

**Unit convention, used throughout.** Capacities in **GiB** (2³⁰ bytes). Context lengths are
binary: **8K = 8,192 tokens, 32K = 32,768, 128K = 131,072**. Model weights quoted in decimal GB
where that is the conventional form, with the GiB equivalent given. Device "80GB" HBM is taken
as 80 GiB.

---

## 0. Four constraints, not one wall

Conflating these produces the false conclusion that adding capacity solves KV scaling.

There are four, and they bind independently:

| Wall | Statement | Effect of adding a memory tier |
|---|---|---|
| **Capacity** | HBM cannot hold the KV of all sequences you want resident | **Relieved.** This is what the tier is for. |
| **Bandwidth (a) — HBM-side** | Attention re-reads the KV of every active sequence at every decode step, out of HBM | **Unchanged.** A property of the attention kernel and HBM, untouched by what lies behind HBM. |
| **Bandwidth (b) — tier-ingress** | KV must cross the tier boundary fast enough to feed attention | **Created.** Did not exist before; now a first-class constraint. |
| **Transfer / latency** | Each individual transfer must complete before compute stalls on it | **Created.** Distinct from (b): average bandwidth can be sufficient while individual transfers still stall. |

The two bandwidth constraints are genuinely different and both bind:

```
        CXL / tier memory
                │
                │  ← tier-ingress bandwidth  (created by tiering)
                ▼
              HBM
                │
                │  ← HBM bandwidth           (unchanged by tiering)
                ▼
        attention kernel
```

The honest architectural claim:

> **A memory tier does not eliminate the KV memory wall. It changes its shape — from a capacity
> wall into a data-movement wall.**

### 0.1 Measured: the HBM-side bandwidth wall is real and steep

Wall (a) is no longer an assertion. On a DGX Spark (GB10, 128 GiB unified LPDDR5X, 273 GB/s
spec), sweeping resident KV volume and fitting decode step time against it:

| Framework | Fit (ms) | n | R² | Weight BW | KV BW |
|---|---|---|---|---|---|
| HuggingFace + SDPA, unpaged | `100.3 + 16.34·KV_GiB` | 7 | 0.991 | 151.5 GB/s | **65.7 GB/s** |
| vLLM + FlashAttention-2, paged | `73.9 + 17.58·KV_GiB` | 9 | 0.945 | 205.7 GB/s | **61.1 GB/s** |

Achieved streaming bandwidth on the same machine is **236.5 GB/s (87% of spec)**. So weights
are read at 64–87% of achievable bandwidth and **KV at 26–28%**.

Two consequences, both measured rather than argued:

- **The single-bandwidth roofline is wrong.** Decode time is two-term:
  `t = W/B_weight + KV/B_kv`, with `B_kv ≈ 3.4× lower` than `B_weight` on this hardware.
- **The kernel is not the lever.** Two frameworks with different attention implementations and
  different KV layouts give KV slopes within **8%** of each other, while the weight read
  improves 34% between them. Swapping kernels fixes the weight path and leaves the KV path
  where it was.

This is the strongest available evidence for the row above: adding capacity behind HBM cannot
touch a constraint that binds on the HBM side, and neither can a better attention kernel. Only
reducing KV bytes read per step does — which is why quantisation, not placement, is the
first-order lever against this particular wall.

That is a real win, because capacity is a hard limit and data movement is a scheduling problem.
But it is a *different* problem, and the rest of this section is about solving that one.

---

## 1. Sizing the capacity wall

```
bytes_per_token = 2 × layers × kv_heads × head_dim × bytes_per_element
                  ↑
                  K and V
```

The decisive term is `kv_heads`, not query heads. Grouped-query attention decoupled them;
using the query-head count overstates modern models by the GQA ratio.

| Model | KV heads | KiB/token | 8K | 128K |
|---|---|---|---|---|
| Llama-2 70B (MHA, fp16) | 64 | 2,560 | **20.0 GiB** | 320 GiB |
| Llama-3 70B (GQA, fp16) | 8 | 320 | **2.50 GiB** | 40.0 GiB |
| Llama-3 70B (GQA, fp8) | 8 | 160 | 1.25 GiB | 20.0 GiB |
| Llama-3 8B (GQA, fp16) | 8 | 128 | 1.00 GiB | 16.0 GiB |

Llama-3 70B at 8K is **2.50 GiB per sequence**, not 20 GB. The 20 GB figure is the pre-GQA
multi-head number and belongs to the previous model generation.

> Independently confirmed: Xu, Khaira & Singh, arXiv:2603.20397, Figure 3 — "70B-GQA
> (0.31 MB/token, 80L×8H)… Values computed as 2 × L × H_kv × d_h × 2 bytes per token."

### 1.1 When does KV actually exceed the weights?

Not at 32K. Llama-3 70B in bf16 is 140 GB = **130.4 GiB** of weights.

```
 32K context →  10.0 GiB KV  ( 8% of weights)
128K context →  40.0 GiB KV  (31% of weights)

single-sequence crossover ≈ 427,000 tokens
```

The crossover is driven by **concurrency**, not context length alone. Computed in consistent
GiB throughout:

```
  8K context: 130.4 / 2.50  ≈ 52 concurrent sequences to match the weight footprint
128K context: 130.4 / 40.0  ≈  3 concurrent sequences
```

On a 4×H100 node (4 × 80 GiB = 320 GiB), holding 130.4 GiB of weights and ~20 GiB of
workspace, **169.6 GiB** remains for KV:

```
  8K: ~68 sequences        128K: ~4 sequences
```

This is a **KV-only bound, not a capacity plan.** Realised concurrency also depends on TP and
KV sharding, generated-token allowance, CUDA graph reservations, allocator fragmentation,
scheduler policy, prefix sharing, speculative decoding, and maximum sequence length.

Defensible summary sentence: *as context length and concurrency grow, aggregate KV memory can
exceed the model-weight footprint.*

---

## 2. Block-granular KV management

Reserving contiguous memory sized to the maximum possible context wastes most of it.

The enabling abstraction is **block-granular KV management**: fixed-size blocks, non-contiguous
physical placement, a logical-to-physical mapping, copy-on-write for shared blocks.
**PagedAttention** (Kwon et al., SOSP 2023) is the canonical implementation, reporting 2–4×
higher throughput at the same latency, losslessly.¹

It is an important enabling abstraction rather than a strict precondition — tiering, offload,
compression and prefix caching can each be implemented without this specific mechanism. But
without *some* block granularity, migration and sharing become far harder, and every technique
below assumes it.

---

## 3. Prefix reuse — conditional on workload

Where many requests share a long identical prefix — a system prompt, a tool-definition library,
a retrieved document — recomputing it per request is waste. **RadixAttention** (SGLang)
organises cached prefixes as a radix tree, matches the longest shared prefix on arrival, and
skips prefill for the matched segment.

For workloads with substantial prefix sharing this is the highest-leverage optimisation
available, because it *deletes* work rather than relocating it. For workloads with little
repeated prefix — diverse single-turn queries, per-user private context — its value approaches
zero. It is not a universal first step; see §6.

---

## 4. Managing state across the hierarchy

### 4.1 The organising distinction: lossless versus lossy

Every technique below either preserves the KV state the model would otherwise have computed, or
discards/approximates part of it.

- **Lossless state management** — the same KV is used, just held somewhere else or moved at a
  different time. (Floating-point results may still differ in the last bits due to kernel
  choice, accumulation order or async scheduling; that is a reproducibility property, not an
  information-loss property, and is not the distinction being drawn.)
- **Lossy state reduction** — some KV is discarded, compressed with error, reconstructed
  approximately, or substituted by a similar block.

**A reviewer will ask: are you proposing a lossless hierarchy, or lossy KV selection?**
This section proposes a **lossless hierarchy**, and treats lossy reduction as an orthogonal,
optional layer that can be composed on top.

### 4.2 Lossless placement

| Keep in HBM | Move to the tier | Basis |
|---|---|---|
| Shared prefixes | — | Highest reuse per byte; lossless by construction |
| KV of the currently-scheduled **decode set** | KV of idle inter-turn sessions | Idle state is not read until the session re-enters the decode set |
| Blocks needed within the prefetch horizon | Blocks beyond it | A scheduling decision, not an approximation |

These are three **foundational** lossless placement decisions, not the entire lossless design
space. Placement can also vary losslessly by layer, head group, tenant, device, NUMA domain,
tensor-parallel shard, pipeline stage, block age, recomputability, prefix ownership, and
execution phase. What the three share is that they carry no accuracy risk and require no
accuracy evaluation.

### 4.3 Lossy reduction — a separate layer with separate obligations

| Technique | What is approximated | Reported cost¹ |
|---|---|---|
| ShadowKV² — low-rank keys resident, values offloaded, landmark retrieval | SVD key reconstruction; chunk-landmark selection | 6× GPU memory reduction, 3.04× throughput; accuracy holds while sparse budget < 1.56% |
| TailorKV³ — quantise shallow layers, offload deep layers, fetch top-k | Layer-dependent quantisation and token selection | ~73.8% GPU memory reduction, 8–18× faster than standard offloading |
| CLO⁴ — head-wise approximate caching, critical heads pinned | Query-similarity reuse of previously loaded KV | 9.3–66.6% throughput over SOTA; ≤0.42 accuracy drop |
| H2O, SnapKV, Ada-KV | Token discarding by attention score | 5–10× memory reduction |

These are structural observations exploited as heuristics — which is what good systems work
looks like — but they are heuristics, they are workload-dependent, and they carry an accuracy-
evaluation burden that lossless tiering does not.

> *Published systems exploit different structural properties of KV state — reuse, layer
> behaviour, head importance, compressibility, predictability — to reduce HBM residency or hide
> transfer cost.*

### 4.4 The hard case: one sequence larger than HBM

At 128K, Llama-3 70B needs 40 GiB for a *single* sequence. A handful of concurrent long-context
requests and "keep the active context resident" stops being a policy.

Streaming is one answer, and it is feasible because attention is decomposable: online-softmax
(FlashAttention-style) accumulation lets attention run over KV chunks without the full context
being simultaneously resident. Two constraints then bind, and **satisfying the first does not
imply satisfying the second**:

```
SUSTAINABILITY   B_tier  ≥  Σ_{i ∈ decode set} R_KV,i
                 aggregate consumption across ALL concurrently decoding sequences,
                 not one sequence in isolation

CONTENTION       Σ_i B_i(t)  ≤  B_tier_available(t)
                 concurrent transfers share the link; this is where per-tenant
                 QoS metadata earns its place

HIDING           T_transfer(chunk)  ≤  T_compute(previous chunk) + prefetch_slack
                 each transfer must complete inside the compute window it hides
                 behind — a per-transfer condition, not an average
```

Average bandwidth can be sufficient while the pipeline still stalls, because of transfer
latency, insufficient prefetch depth, burstiness, queue contention, synchronisation, controller
scheduling, DMA setup, competing sequences, and head/layer traversal order.

Techniques in this section act on one of three terms: **raising `B_tier`** (wider link, better
controller), **lowering the consumption rate** (quantisation, sparsity, fewer resident
sequences), or **widening the hiding window** (deeper prefetch, chunk sizing, reordering).

### 4.5 Movement amplification

Hit rate is the wrong headline metric: a placement controller can show a high HBM hit rate while
thrashing blocks across the boundary.

**KV blocks are write-once.** For a given (sequence, layer, position range), K and V are computed
once and never modified. So a tier copy never goes stale, evicting a block that already has one is
a clean discard, and each block is therefore demoted **at most once** however many times it is
promoted. This is a real difference from a general-purpose cache and it fixes the floor.

With `U` = unique payload bytes required, `n_b` promotions and `δ_b ≤ 1` demotions:

```
A_read     =  Σ_b n_b·M_b / U              promotion traffic per useful byte
A_traffic  =  Σ_b (n_b + δ_b)·M_b / U      total boundary traffic per useful byte

never leaves HBM            δ=0, n=0  →  0
written out, never recalled δ=1, n=0  →  1
written out, recalled once  δ=1, n=1  →  2   ← floor for any block that round-trips
written out, recalled 3×    δ=1, n=3  →  4   ← thrashing
```

The floor for a round-tripping block is **2, not 1**. No universal threshold is asserted; the
comparison is against the workload's own lower bound, obtained by replaying the trace under
offline-optimal placement.

**Reuse per moved byte** is the corresponding economic objective. A block may cross the boundary
once and be read by attention hundreds of times from HBM:

```
                Σ_b (attention reads of b while resident)·M_b
reuse ratio =  ──────────────────────────────────────────────
                        Σ_b (n_b + δ_b)·M_b
```

A block that round-trips once and is read 500 times has a reuse ratio of 250. Maximising useful
HBM reuse per tier byte moved is a better policy objective than hit rate, because hit rate is
blind to what the hit cost to arrange.

**Byte-time, not bytes, is the scarce commodity.** 5 MiB held for 1 ms and 5 MiB held for 500 ms
are not the same allocation. This is what puts idle inter-turn state and active decode state on
one scale — see `formal_core.md` §3.

### 4.6 Movement versus regeneration

Three quantities with three different epistemic statuses.

| Quantity | Value | Status |
|---|---|---|
| Transfer of one 16-token block (5.00 MiB) | **43 µs @ 121 GB/s · 82 µs @ 64 GB/s · 164 µs @ 32 GB/s** | Idealised lower bounds across the plausible effective-bandwidth range (Appendix A). Full link utilisation, no queueing. |
| Forward compute for 16 tokens, linear term only | **9 ms @ 50% MFU · 11 ms @ 40% · 18 ms @ 25%** | First-order estimate, strongly MFU-dependent. |
| Regeneration of an arbitrary *interior* block | **≥ the above, potentially far higher** | Interior KV is not independently reproducible — hidden states depend recursively on all preceding context through every layer. Reconstruction may require re-prefilling a substantial prefix. |

Across the whole plausible range, moving a block is **two to three orders of magnitude** cheaper
than regenerating it. Claim that; do not claim a precise multiple, because the two quantities do
not measure the same operation.

Note also that this is not binary. **KVPR**⁸ profiles link bandwidth against compute throughput
and deliberately recomputes a *fraction* of the KV while transferring the remainder, overlapping
the two. The split is a tunable.

And "moving a block" is a per-transfer cost, not a per-block cost: total movement cost is
governed by amplification (§4.5), not by a single transfer.

### 4.7 What the surveyed systems optimise

| System | Category | Mechanism | Reported¹ |
|---|---|---|---|
| PagedAttention⁵ | GPU memory management | Paged allocation, prefix sharing | 2–4× throughput |
| LayerKV⁹ | Placement | Layer-wise residency; offload time ≤ prefill time | up to 69× TTFT |
| InfiniGen⁶ | Prefetch | Speculative prefetch via query similarity + SVD | 1.63–32.9× |
| INF²⁷ | Near-storage compute | Attention on CSD-attached FPGAs | 3.46× throughput |
| KVPR⁸ | Recompute/transfer split | Profiled partial recomputation, overlapped | 35.8% lower latency |
| Oneiros¹⁰ | Memory reclamation | Parameter remapping from idle models | 20.7–99.3% TTFT reduction |
| CLO⁴ | Approximate caching | Head-wise caching, zero-copy, similarity reuse | 9.3–66.6% throughput |

These are not seven comparable systems — they range from a GPU allocator to a near-storage
accelerator — and must not be read as a controlled comparison.

### 4.8 From eviction policy to admission-aware placement

> The surveyed systems place greater emphasis on placement, transfer overlap, prefetch,
> compression and scheduling than on conventional cache replacement. This suggests — but does not
> establish — that replacement policy may be a secondary optimisation in many tiered-KV
> deployments.

Under a binding capacity constraint admission and eviction are **dual** — admitting a block when
HBM is full forces another out — so eviction does not disappear. The claim is subordination, not
elimination:

> Conventional eviction asks which resident object should leave. A tiered KV controller begins one
> level higher: **which state deserves HBM residency over the upcoming scheduling horizon?**
> Eviction becomes a consequence of that decision rather than an independent policy.

| Classical cache asks | A tiered KV system asks |
|---|---|
| What should I evict? | Which sequence's blocks should be admitted to HBM, **at what time**, **for how long**, and **at whose expense**? |

That moves the problem toward scheduling theory rather than cache-policy folklore, and it
composes with the serving scheduler rather than fighting it.

Two supporting observations, offered as reasoning rather than evidence:

**Access is partially announced.** Once a sequence is in the decode set, its existing context is known —
genuinely more information than a general-purpose cache has. But the *future* schedule is not
known: continuous batching, preemption, request arrival, speculative-decoding acceptance,
sequence completion, admission under memory pressure, and multi-tenant interference all mutate
it. KV access is *more predictable* than general cache access; it is not announced.

**Under dense attention, "which blocks" is not the question** — every block of an active sequence
is read every step. The open questions are *when* each must be resident, *how far ahead* to
transfer, *how much* transfer overlaps compute, and *which competing sequence* gets scarce HBM.
The qualifier matters: "which blocks" becomes live again the moment attention becomes sparse —
that is, the moment you cross from §4.2 into §4.3. Whether replacement policy matters is
*downstream of* the lossless/lossy choice, not independent of it.

---

## 5. Quantisation

| Method | Compression | Cost¹ |
|---|---|---|
| KIVI (2-bit, per-channel keys / per-token values) | 2.6× peak memory; 2.35–3.47× throughput | <2% accuracy drop on most models |
| KVQuant (3-bit, non-uniform, pre-RoPE) | 3.7–6.9× memory | <0.1 perplexity degradation |
| PALU (low-rank projection) | ~50% KV compression | 1.89× speedup (2.91× with quantisation) |
| MiniCache (cross-layer merge) | up to 41% memory | minimal loss |

Quantisation is one of the few techniques that acts on **three** of the walls in §0 at once: it
reduces capacity demand, reduces HBM-side bytes read per decode step, and reduces tier-ingress
bytes. That breadth, not the compression ratio alone, is why it ranks where it does.

The measurement in §0.1 sharpens this. Because KV bytes are read at ~26% of achievable
bandwidth while weight bytes reach 87%, **a byte removed from the KV cache is worth ~3.4 bytes
removed from anywhere else** on GB10. Halving KV precision is therefore worth materially more
than its 2× byte reduction suggests. This is a hardware-dependent claim: on a machine where
KV and weight bandwidth are closer, the multiplier shrinks. It should be measured per platform,
not assumed.

KVQuant's pre-RoPE detail is worth noting: RoPE is a rotation applied to keys, and quantising
*before* the rotation is measurably easier than after. That is what a genuine
position-encoding-aware optimisation looks like — RoPE constrains *how you compress*, not *what
you prefetch*.

---

## 6. A decision procedure

```
Do requests share substantial prefixes?
  └─ yes ──► prefix reuse (radix/prefix caching)     ← deletes work; do this first
     ▼
Is reservation/fragmentation waste significant?
  └─ yes ──► block-granular allocation               ← lossless
     ▼
Does aggregate KV exceed HBM?
  └─ no  ──► stop. You do not need a tier.
     ▼
Does a SINGLE active sequence exceed HBM?
  └─ yes ──► exact single-device execution now requires EITHER streaming/tiering
  │          OR more memory. Otherwise you must change an execution, model, or
  │          accuracy constraint: cap the context, reject the request, shard KV
  │          across devices, quantise, sparsify attention, or change model.
  │          Streaming is one architectural answer, not the mandatory one.
     ▼
Is the accuracy budget non-zero?
  └─ yes ──► lossy reduction becomes available (§4.3)
  └─ no  ──► lossless tiering only
     ▼
Does the pipeline stall despite adequate average bandwidth?
  └─ yes ──► the HIDING constraint binds, not SUSTAINABILITY (§4.4):
  │          deepen prefetch, resize chunks, reorder traversal
     ▼
Is movement amplification materially above the workload's own lower bound? (§4.5)
  └─ yes ──► placement/admission policy is thrashing; fix that before
  │          tuning replacement
  └─ no  ──► replacement policy is second-order
```

---

## 7. Where a contribution could sit

### 7.1 Scoping, stated carefully

Among the eleven systems reviewed in this section, I did not identify one that (a) targets a
CXL 3.0 memory tier with an on-device controller, or (b) specifies what happens to
cache-management metadata when a block migrates between tiers. **This is a scoping observation
over a limited review, not a claim of novelty.** Establishing novelty requires a systematic
literature review that this section does not constitute.

### 7.2 The idea, stated at the right level

"Preserve metadata when KV moves" is close to obvious once said. The stronger formulation:

> **Make KV state and its control metadata a single migratable object.**

```
                     KV BLOCK OBJECT
        ┌────────────────────────────────────────┐
        │  K / V payload                         │
        ├────────────────────────────────────────┤
        │  identity / prefix hash                │
        │  tenant + QoS class                    │
        │  reuse history, recency                │
        │  compression representation            │
        │  prefetch + residency state            │
        │  provenance / sequence association     │
        └────────────────────────────────────────┘
                          │
                  MIGRATES AS A UNIT
                          ▼
        ┌───────────┐            ┌───────────────┐
        │  GPU HBM  │ ◄────────► │  Tier memory  │
        └───────────┘            └───────────────┘
```

Payload-capacity overhead is negligible — 64 B against a 5.00 MiB block is **0.0012%**. Metadata
*processing* overhead is a separate matter and is not established: lookup bandwidth on the
promotion path, atomic update under concurrent scheduler writes, controller SRAM for the resident
index, policy-state coherence between planes, and per-transaction link cost all remain to be
measured. Whether persistent policy state changes outcomes is a **testable hypothesis, not a
claim**.

### 7.3 The question that decides whether this is an architecture

> **Why must the controller be on the device?**

"Because that is where the memory is" is not an answer. A defensible answer must name something
device residency buys that the host runtime cannot obtain as cheaply:

- eliminates host round-trips on promotion decisions
- operates on tier-resident metadata at memory latency rather than across the link
- coordinates promotion without CPU involvement, freeing host cycles under multi-tenancy
- maintains importance history across HBM eviction and re-admission, which host runtimes discard

**State the null hypothesis and defeat it.** The host runtime already possesses scheduler intent
— arguably more of it than the device does. So the burden is precise: what does the device know,
or do faster, that the host does not? If none of the four survives scrutiny, the contribution is
the **migratable-object protocol**, and CXL is merely where it happens to run. That would still
be a contribution; it would just be a different one.

### 7.4 The research question

> Can a KV control plane that carries state metadata across tier boundaries, and consumes
> scheduler intent, reduce HBM residency and movement amplification while keeping data movement
> hidden behind attention computation?

Measurable quantities, all obtainable on host-DRAM-over-PCIe before any CXL hardware is
procured:

| Metric | Why it matters |
|---|---|
| HBM residency required for a target QoS | The capacity claim |
| Tier bytes transferred per generated token | The movement claim |
| **Movement amplification** (§4.5) | Whether placement is thrashing |
| Prefetch accuracy / stall time per token | Whether the HIDING constraint is met |
| TTFT, inter-token latency, throughput | What users experience |
| Controller and host CPU overhead | Whether the mechanism pays for itself |
| Fairness under multi-tenancy | Whether admission control is sound |

---

## Appendix A — Assumptions and provenance

Two inputs to the §4.6 economics are chosen, not measured. Stated explicitly, with sensitivity.

| Input | Value | Basis | Sensitivity |
|---|---|---|---|
| Device memory bandwidth (GB10) | **236.5 GB/s measured** | Streaming copy kernel, 4 GiB buffers, median of 20. 87% of the 273 GB/s spec. Supersedes the previous ~50% haircut assumption, which was too pessimistic. | — |
| KV read bandwidth (GB10) | **61.1–65.7 GB/s measured** | Slope of decode step time against resident KV volume, two frameworks (§0.1). | Sets the 3.4× KV-byte penalty |
| Tier effective bandwidth (CXL) | 32 / **64** / 121 GB/s as a range | CXL 3.0 uses the PCIe 6.0 PHY at 64 GT/s; x16 gives ~121 GB/s per direction theoretical. **Still an assumption — no CXL hardware was measured.** | Block transfer 43–164 µs across the range |
| Prefill throughput | 25% / 40% / **50%** MFU presented as a range | H100 SXM BF16 dense tensor peak ≈ 495 TFLOP/s. Published prefill MFU commonly reaches 40–50%. | 16-token estimate 9–18 ms |
| Model geometry | 80 layers, 8 KV heads, head_dim 128 | Published Llama-3 70B configuration | Exact |
| Weight footprint | 140 GB = 130.4 GiB (bf16) | 70e9 × 2 bytes | Exact |
| Device capacity | H100 "80GB" taken as 80 GiB | Convention stated in the preamble | — |

The order-of-magnitude conclusion in §4.6 survives the full range in both dimensions; no precise
multiple is claimed anywhere, and the body quotes ranges rather than a single figure.

**Provenance discipline.** Two measurement bugs were found and fixed while producing §0.1, both
documented in `measured_results.md` §5. The instructive one produced an apparent KV bandwidth of
342.9 GB/s — *above* the 273 GB/s memory bus — caused by identical batch token ids letting
vLLM's prefix cache share one copy of the KV. It was caught only by checking the result against
a physical bound, not by inspection. Any measured bandwidth exceeding the bus is a bug, always.

---

### Sources

1. Xu, Y., Khaira, N. K., & Singh, T. (2026). *KV Cache Optimization Strategies for Scalable and
   Efficient LLM Inference.* arXiv:2603.20397. Tables 4 and 6, Figure 3. Figures marked ¹ are as
   reported in that survey — secondary reporting, not independently verified here.
2. Sun, H., et al. (2025). *ShadowKV.* arXiv:2410.21465.
3. Yao, D., et al. (2025). *TailorKV.* arXiv:2505.19586.
4. Yi, J., et al. (2025). *CLO.* arXiv:2511.14510.
5. Kwon, W., et al. (2023). *PagedAttention.* SOSP. arXiv:2309.06180.
6. Lee, W., et al. (2024). *InfiniGen.* arXiv:2406.19707.
7. Jang, H., et al. (2025). *INF²: Near-Storage Processing.* arXiv:2502.09921.
8. Jiang, C., et al. (2025). *KVPR.* arXiv:2411.17089.
9. Xiong, Y., et al. (2024). *LayerKV.* arXiv:2410.00428.
10. Li, R., et al. (2025). *Oneiros.* arXiv:2507.11507.
11. Lee, D., et al. (1999). *On the Existence of a Spectrum of Policies that Subsumes the LRU and
    LFU Policies.* ACM SIGMETRICS. (Any decayed-frequency score is LRFU; α is its documented
    LRU↔LFU spectrum parameter.)
