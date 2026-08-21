# RETRACTIONS

## What happened

Version 4.0 of this package presented an internally consistent set of headline numbers — a 97% HBM
hit rate and everything derived from it — that had been **constructed rather than measured**, and
labelled as an analytical model in the fine print while being quoted as a result in the headlines.
A hardware measurement on a DGX Spark (GB10) on 19 August 2026 and a corrected trace-driven
simulation contradict them, and they are withdrawn.

This file is the complete record. It lists every retracted claim, what each rested on, what
contradicted it, and what replaces it; it then records a second retraction *inside* the simulation
itself, two measurement bugs found while producing the v5.0 data, and the pattern all of them share.

---

## 1. Retracted claims

| v4.0 claim | What it rested on | What contradicted it | What replaces it |
|---|---|---|---|
| **97% HBM hit rate** | The additive ladder below, applied to the S1 scenario | Corrected simulation (Rev 2, `results_v2.json`) | **8.52–91.59%**, depending on HBM budget, policy and workload. There is no single number to quote. |
| **The 72 → 80 → 87 → 93 → 97 ladder** (+8 / +7 / +6 / +4 pts for anchor pinning, EMA α=0.2, per-head tracking, RoPE-aware prefetch) | Nothing. No ablation was ever run | Never reproduced; the increments were assumed additive and independent, and the mechanisms they decompose overlap | Withdrawn without replacement. Per-technique ablation remains unrun. |
| **+25 pts over LRU** (full stack) | The ladder's endpoints, 97 − 72 | Rev 2 policy sweep and α grid | **+2.5 to +5.8 pts** over LRU at the specified α (experiment `e1`); **+19.1 pts** at a tuned α ≈ 0.05 (experiment `e2`). Different experiments, different baselines — do not mix rows between them. |
| **16× user capacity** | Downstream of the 97%: a 2.31 GB per-user hot set out of 37 GB of evictable HBM | The hit rate it depended on; and the correctness constraint — under dense attention every block of an active sequence is read every step, so there is no 5.4% subset to be resident | Withdrawn. Concurrency bounds are stated as KV-only capacity arithmetic, not as a served-user count. |
| **5.4% hot set captures 97% of accesses** | Reading a cache hit rate and attention sparsity as the same quantity | They are two different claims with different obligations; the formal correctness constraint separates them | Withdrawn. Lossless residency is `x_b(t) = 1` for all `b ∈ Read(t)`; sparsity is a *lossy* relaxation of that constraint and carries an accuracy-evaluation burden. |
| **65× single-access latency** (13.0 µs ÷ 200 ns) | A component sum for a hypothetical load/store path divided by a component sum for a driver-mediated DMA path | No CXL hardware was measured; the two paths are not the same operation | Retracted as a headline. Tier access is a range, not a ratio. |
| **200 ns CXL access** | A six-line component budget (issue, PHY, switch, decode, DDR5, return) | Unmeasured on any device | **Demoted** to an unmeasured assumption inside a 32–121 GB/s effective-bandwidth range; a 5.00 MiB block transfer is 43–164 µs across that range. |
| **11.5× effective end-to-end latency** | `0.97 × 100 ns + 0.027 × 200 ns + 0.003 × 25 µs` — the 97 / 2.7 / 0.3 tier split | Derived from the 97% | Withdrawn with it. |
| **15.6× TTFT** on a resumed 128K request | 43 GB ÷ 16.4 GB/s baseline against 43 GB ÷ 256 GB/s proposed | The 16.4 GB/s baseline assumption below | Withdrawn. |
| **36% CapEx reduction** ($160,000 → $103,000) | Pricing two configurations both assumed to serve 16 users at 128K | It priced a configuration justified by the 16× | Withdrawn. No cost model is restated in v5.0. |
| **16.4 GB/s host-DRAM offload baseline** | An undefended estimate of PCIe Gen5 ×16 after protocol overhead and contention | Measured achievable bandwidth of **236.5 GB/s = 87% of spec** on GB10 shows the class of pessimistic-haircut estimate it belongs to is unreliable | Withdrawn as an input. Baselines must be measured, not estimated. |
| **~50% "effective bandwidth" haircut** applied to spec figures throughout | Convention, not measurement | Streaming copy, 4 GiB buffers, median of 20: **236.5 GB/s against a 273 GB/s spec** | Retired. The haircut was wrong — and wrong in the *pessimistic* direction, which is why it survived unexamined. |
| **5–10× preprocessing offload** (tokenisation, image decode, on endpoint ARM cores) | A design target for a device that does not exist | Never measured; no endpoint, FPGA or silicon exists | Withdrawn as unsupported. |

---

## 2. What survives, and why

| Claim | Why it stands |
|---|---|
| **320 KiB/token** for Llama-3 70B GQA (80 layers, 8 KV heads, head_dim 128, fp16) | `2 × layers × kv_heads × head_dim × bytes` — arithmetic over published model geometry. Independently confirmed against a third-party survey. |
| **5.00 MiB blocks** (16 tokens) | 320 KiB × 16. Definitional. |
| **The capacity table** (8K → 2.50 GiB, 128K → 40.0 GiB, crossover ≈ 427,000 tokens) | The same arithmetic over stated inputs. |
| **6× memory expansion** | `(192 + 1,024) ÷ 192 = 6.3×` over stated capacities. |
| **The four-constraint decomposition** (capacity relieved; HBM-side bandwidth unchanged; tier-ingress and per-transfer latency both created) | An analytic decomposition. The measurement in v5.0 §1 *strengthens* it — it confirms the HBM-side row directly. |
| **The write-once property** of KV blocks, and the `A_traffic = 2` floor that follows | A property of autoregressive decoding, not of any implementation. |

The common thread: **each of these is analytic or definitional, and none of them ever depended on
the hit rate.** That is why the retraction is severe but bounded. What fell was the layer of claims
built on a single unverified number; what stands is the layer underneath it.

---

## 3. The second retraction, inside the simulation

The corrected numbers above come from Revision 2 of the tiering simulator. Revision 1 was also
wrong, and its findings are also withdrawn.

**The defect.** `_pick_victim()` re-inserted a popped heap entry only when the recomputed score
**exceeded** the stored key. That is correct for LFU, where counts rise and stored keys are
stale-*low*. It is wrong for a decaying EMA, where scores fall and stored keys are stale-*high*:
the popped entry was returned as the victim without checking whether another block had decayed
further. Audited against a brute-force `argmin`:

| Policy | Wrong victims | Evictions | Rate |
|---|---|---|---|
| EMA α=0.15 | 114,114 | 114,528 | **99.6%** |
| EMA α=0.01 | 73,650 | 74,987 | **98.2%** |
| LFU | 0 | 74,987 | 0.0% |

The fix is an exact order-preserving key, `K = ln S_touch + t_touch·(−ln(1−α))`, which orders
identically to the decayed score at any time; re-audit found 0 wrong of 78,860.

**It inverted four of five findings.**

| Revision 1 finding | Status | Revision 2 |
|---|---|---|
| EMA α=0.15 worst at every budget | **Reversed** | EMA α=0.15 *best* at 4 of 5 budgets (44.18% vs LRU 38.34% @ 32 GiB) |
| Hit rate monotone as α → 0, converging onto LFU | **Withdrawn** | Non-monotonic, interior optimum at α ≈ 0.05 (**55.96%**); no convergence onto LFU |
| LRU wins the scan workload | **Withdrawn** | EMA α=0.01 leads scan (36.26% vs 31.51%) |
| EMA wins the cyclic sweep 2.2× | **Withdrawn** | Only LFU survives loop (22.06%); EMA tracks LRU at 9.83% |
| 97% of stall is recompute; policy nearly irrelevant | **Withdrawn** | Warm recompute is **44–89%** of stall across budgets and policies |
| 8K Llama-3 70B KV is 2.50 GiB, not 20 GB | **Stands** | Unchanged — analytic, simulator-independent |

Revision 1's simulator is retained **unmodified** in `scripts/`, as
`kv_tiering_sim_rev1_retained_for_audit.py`, so the defect and its effect on the published numbers
remain independently checkable rather than merely described. Its superseded output is likewise kept
as `data/sweep_rev1_superseded.json`.

**The consequence for Revision 2.** A single data-structure defect inverted four of five findings.
That is the correct reason to treat Revision 2 as **provisional, not settled**: it has no Belady
bound, no bandwidth or queueing model, it assumes whole-context residency, it uses a synthetic
workload over 2–3 seeds, and it has no hardware validation. Every table drawn from it in this
package carries that caveat.

---

## 4. Two measurement bugs

Both were found while producing the v5.0 hardware data, and both are recorded because both produced
plausible-looking wrong answers.

**1. Silent fp32 load.** `torch_dtype=` is deprecated in current transformers and was silently
ignored, loading weights in fp32 (28.4 GiB) and triggering the OOM killer. Unified memory surfaces
a CUDA OOM as a Linux OOM kill, which obscures the cause entirely. Fixed with `dtype=`; the script
now prints resident weight bytes and dtype so a repeat is visible.

**2. Prefix-sharing artefact.** The first vLLM run generated identical token ids for every sequence
in the batch. vLLM's prefix cache stored **one shared copy** of the KV, so true resident KV was
`L × bpt` rather than `B × L × bpt` — inflating the x-axis by a factor of B and producing an
apparent KV bandwidth of **342.9 GB/s**. That is above the 273 GB/s memory bus, which is what
exposed it. **Any measured bandwidth above the bus is a bug, always.** Fixed by giving each
sequence distinct tokens and disabling prefix caching.

The second is the one to remember. It did not produce an obviously broken number — it produced a
*better* number, one that would have supported a more exciting conclusion ("vLLM is 5× more
efficient on KV"). **It was caught by a physical bound, not by inspection.** Nothing about reading
the code would have flagged it, and nothing about the shape of the result invited suspicion.

---

## 5. What the pattern says

Every error in this record ran in the same direction. The ladder was additive because additive was
tidier and larger. The hit rate was 97% rather than 80%. The bandwidth haircut was pessimistic on
the baseline and generous on the proposal. The heap defect made the *proposed* policy look worst,
which read as commendable conservatism until the correction showed it was simply wrong. The prefix
bug produced a bandwidth above the memory bus in support of a more exciting conclusion.

That is not coincidence, and it is not dishonesty either. It is what unverified construction does:
when a number is chosen rather than observed, the choice is made by someone who already knows which
answer would be more interesting, and there is no mechanism in the process to say no. The only
mechanisms that ever said no here were a physical bound and a brute-force audit — and both are
cheap. Neither required new hardware.

Two practical conclusions follow, and they are the reason this file exists rather than a quiet
revision. **Measure early**, because the first measurement retired two years of assumption in a day
and cost one machine and one afternoon. **Publish the scripts**, because the defect that inverted
four findings was found by running an audit against the published code, and the code is retained
unmodified so anyone can run it again.

---

## 6. Files still in circulation

Two artefacts distributed before the erratum are **Revision 1** outputs and should not be relied on:

- **`findings.md`** — the Revision 1 policy findings, four of five of which are inverted above.
- **`kv_policy_results.html`** — the Revision 1 results page.

They are superseded by **`data/ERRATUM.md`** and **`data/results_v2.json`**, and by
`CANONICAL-NUMBERS.md` §3 where the spec and the erratum differ (the spec is computed directly from
`results_v2.json` and is authoritative; the erratum quotes 55.93% for the α optimum against the
spec's 55.96%, and 50–67% for warm recompute against the spec's 44–89%, because the erratum was
written from a narrower slice).

Both circulating files read as finished, and they are wrong. If you are holding a copy, that is the
copy to discard.
