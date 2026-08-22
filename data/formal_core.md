# Formal Core

Three things locked down: the **object model**, the **control/data-plane split**, and the
**objective function**. No new prose argument; this is the specification the chapter's prose
should serve. Intended to sit as an appendix to §7, or to become §8.

---

## 0. One property the rest depends on

**KV blocks are write-once.** For a given (sequence, layer, position range), K and V are
computed once — during prefill, or at the decode step that produces that token — and are never
subsequently modified. Nothing in autoregressive decoding mutates existing KV.

Consequences, all of which distinguish this from a general-purpose cache:

- A tier copy, once written, **never becomes stale**. There is no coherence problem on the
  payload.
- Eviction of a block that already has a tier copy is a **clean discard** — no writeback.
- Therefore each block is demoted **at most once** in its lifetime, regardless of how many times
  it is promoted.
- The only mutable state associated with a block is **control metadata**, which is orders of
  magnitude smaller than the payload and is handled separately (§1).

Exception to state explicitly: in-place re-quantisation or re-compression of a resident block
writes new payload bytes and breaks write-once for that block. Systems that do this must account
for it.

---

## 1. KV object model

### 1.1 The object

A **KV block object** is the unit of placement, migration and accounting.

```
KV BLOCK OBJECT  b
├─ PAYLOAD            K/V tensors for (sequence, layer, position range)   [write-once]
├─ IDENTITY   (immutable)
│    sequence id · position range · layer · head group
│    prefix hash · provenance / parent block
├─ POLICY     (mutable in place, WITHOUT migrating the payload)
│    tenant · QoS class · priority · reuse score · recency
│    scheduler-supplied residency intent
└─ MECHANISM  (mutable, owned by whichever plane holds the block)
     residency location · compression format · transfer state
     validity · in-flight/pinned flags
```

### 1.2 What migrates, and what does not

This is the correction to "migrate metadata with the block." Migrating *all* metadata as one
physically inseparable unit is wrong, because **policy metadata changes while the block does
not move** — QoS is reassigned, priority shifts, scheduler intent updates every step.

| Class | Lifetime | Migrates with payload? | Mutable without migration? |
|---|---|---|---|
| Identity | Fixed at creation | Yes — it *is* the block's name | No |
| Policy | Changes on scheduler events | **Logically bound, physically separable** | **Yes** |
| Mechanism | Changes on every transfer | No — regenerated per location | Yes |

So the design claim is not "one physically fused object." It is:

> **A logically migratable KV object with independently updatable control metadata.**

Identity travels with the payload. Policy is *associated* with identity and may be updated
in place, on either side, without moving a byte. Mechanism is local to wherever the block
currently sits.

### 1.3 What is actually being proposed

Today, in the systems reviewed, a block evicted from HBM loses its accumulated policy state and
is re-admitted as if newly created. The proposal is that **identity-keyed policy state survives
the round trip**. That is a protocol claim, not a hardware claim, and it is testable without any
CXL device.

### 1.4 Overhead — what is and is not negligible

Payload-capacity overhead is negligible: 64 B of metadata against a 5.00 MiB block is 0.0012%.

**Metadata processing overhead is not established and must be measured.** Open items: lookup
bandwidth on the promotion path, atomic update cost under concurrent scheduler writes,
controller SRAM footprint for the resident index, coherence of policy state between planes,
and per-transaction overhead on the link. None of these is bounded by the capacity argument.

---

## 2. Control plane / data plane

### 2.1 The split

The null hypothesis is strong and should be stated first: **the host serving runtime knows more
than any device-side controller can.** It holds the active decode set, next-step candidates,
tenant priorities, preemption decisions, sequence completion, speculative-decoding acceptance,
and admission state. A CXL controller has none of that intrinsically.

The defensible decomposition follows directly:

> **Host owns policy. Device owns mechanism.**

```
              SERVING SCHEDULER
                     │  execution intent: decode set, priorities,
                     │  prefetch horizon, QoS, preemption
                     ▼
        ┌──────────────────────────────┐
        │   KV CONTROL PLANE  (host)   │
        │   admission · residency      │
        │   prefetch horizon · QoS     │
        │   representation choice      │
        └──────────────┬───────────────┘
                       │  placement intent (block ids + deadlines + priority)
                       ▼
        ┌──────────────────────────────┐
        │   KV DATA PLANE  (device)    │
        │   promotion · demotion · DMA │
        │   transfer scheduling        │
        │   compression execution      │
        │   metadata maintenance       │
        └──────────────┬───────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
       GPU HBM                 Tier memory
          │                         │
          └────────► metrics ◄──────┘
                       │
                       ▼
               control feedback
```

The device does not decide which tenant deserves HBM. It is told priorities and deadlines and
decides **how** to execute the resulting movement efficiently.

### 2.2 What would justify device residency — as falsifiable claims

Each of these is a hypothesis with a measurement attached. None is assumed true.

| Claim | Measurement that would support it | Measurement that would refute it |
|---|---|---|
| Device-local decisions remove host round-trips from the promotion path | Promotion latency with device-side scheduling vs host-issued DMA, under load | Host-issued path already hidden behind compute; no exposed stall difference |
| Metadata lookup at memory latency matters | Fraction of promotion latency spent in metadata lookup | Lookup is off the critical path or is trivially cacheable in host DRAM |
| Frees host CPU under multi-tenancy | Host CPU utilisation attributable to KV movement, at scale | CPU is not the bottleneck at realistic tenant counts |
| Sustains policy state across eviction/re-admission | Hit rate and movement amplification, with and without persistent policy state | Host-side persistence achieves the same result |

**The fourth is the only one that does not require the controller to be on the device.** If it
is the one that survives, the contribution is the object model and protocol of §1, and CXL is
simply where it runs — which is still a contribution, just a different one.

---

## 3. Objective function

### 3.1 Decision variables

For each block `b` and time step `t`:

```
x_b(t) ∈ {0,1}        block b resident in HBM at t
p_b(t) ∈ {0,1}        promotion of b initiated at t   (tier → HBM)
d_b(t) ∈ {0,1}        demotion of b initiated at t    (HBM → tier)
```

`M_b` = payload bytes of block b. `Δt` = step duration.

### 3.2 Correctness constraint — the line between lossless and lossy

Let `Read(t)` be the set of blocks attention reads at step `t`.

```
LOSSLESS:   x_b(t) = 1   for all b ∈ Read(t), for all t          (hard)
```

**This single constraint is the formal content of the lossless/lossy distinction.** Lossy
methods — token eviction, sparse retrieval, approximate reconstruction — operate by *relaxing*
it: they permit `b ∈ Read(t)` with `x_b(t) = 0`, substituting an approximation. Everything else
in the model is common to both.

Under dense attention, `Read(t)` is the entire context of every sequence in the decode set,
which is why "which blocks" is not the interesting question and *when* and *at whose expense*
are.

### 3.3 Resource constraints

```
CAPACITY        Σ_b M_b · x_b(t)  ≤  C_HBM                       ∀t

LINK            Σ_b M_b · (p_b(t) + d_b(t)) / Δt  ≤  B_tier(t)   ∀t

SUSTAINABILITY  Σ_{i ∈ decode set} R_KV,i  ≤  B_tier
                aggregate KV consumption rate across ALL concurrently
                decoding sequences, not one sequence

HIDING          T_transfer(chunk)  ≤  T_compute(previous chunk) + prefetch_slack
                per transfer, not on average

QoS             TTFT_j ≤ τ_j,   ITL_j ≤ ι_j                      ∀ tenants j
```

Sustainability and hiding are independent: the first can hold while the second fails, producing
stalls at satisfactory average bandwidth. Both must be stated because techniques act on
different terms — link width and compression raise or relieve the first; prefetch depth, chunk
sizing and traversal order address the second.

### 3.4 Objective

The workload and its QoS targets are the **constraint**; resources consumed are the
**objective**. Minimising HBM occupancy for its own sake would be degenerate — the point is to
serve the same work with less.

```
minimise    Σ_t Σ_b  M_b · x_b(t) · Δt                    ← HBM byte-time
     + λ ·  Σ_t Σ_b  M_b · (p_b(t) + d_b(t))              ← tier traffic
     + μ ·  Σ_t       stall_exposed(t)                     ← unhidden transfer

subject to  the constraints of §3.2–§3.3
```

**Byte-time is the scarce commodity, not bytes.** 5 MiB held for 1 ms and 5 MiB held for 500 ms
are not the same allocation. This is what makes idle inter-turn state and active decode state
comparable on one scale, and it is why the problem is temporal resource allocation rather than
caching.

### 3.5 Admission, and where eviction sits

Under a binding capacity constraint, admission and eviction are dual: admitting `b` when HBM is
full forces some `b'` out. Eviction does not disappear.

> Conventional eviction asks which resident object should leave. A tiered KV controller begins
> one level higher: **which state deserves HBM residency over the upcoming scheduling horizon?**
> Eviction becomes a consequence of that decision rather than an independent policy.

Formally, admission over a horizon `H` with per-block value `V_b` (from QoS, expected reuse,
and scheduler intent) and residency cost `M_b · H`:

```
maximise   Σ_b V_b · x_b        subject to   Σ_b M_b · x_b ≤ C_HBM
```

— a knapsack over the decode set, re-solved as the set changes. Replacement policy is what
remains once this is fixed, which is the formal statement of "subordinate to admission."

### 3.6 Terminology

Continuous batching means there is no *currently scheduled sequence*. There is a
**currently scheduled decode set**, membership of which changes step to step. All residency
language should use the set, not the singular.

---

## 4. Metrics

### 4.1 Movement amplification — corrected

The Rev 3 definition conflated directions and asserted a floor of 1.0, which is wrong. Two
metrics, plus the write-once bound from §0.

Let `U` = unique payload bytes the computation requires. For block `b`, let `n_b` = number of
promotions and `δ_b` = number of demotions.

```
A_read     =  Σ_b n_b · M_b  /  U            promotion traffic per useful byte
A_traffic  =  Σ_b (n_b + δ_b) · M_b  /  U    total boundary traffic per useful byte
```

**Write-once gives `δ_b ≤ 1`** (§0): once a tier copy exists it stays valid, so later evictions
are clean discards. Therefore:

```
block never leaving HBM          δ=0, n=0  →  contributes 0
block written out, never recalled δ=1, n=0  →  contributes 1
block written out, recalled once  δ=1, n=1  →  contributes 2   ← floor for any block that round-trips
block written out, recalled 3×    δ=1, n=3  →  contributes 4   ← thrashing
```

So `A_traffic = 2` is the floor for round-tripping blocks, not 1. Values materially above the
workload's own lower bound indicate the placement policy is thrashing — which presents
identically to a poor replacement rule in hit rate, but has a different fix.

**No universal threshold is asserted.** The Rev 3 decision tree contained a `1.5` that nothing
in the document established; it is removed. The comparison is against the workload-specific
lower bound, computed by simulating the same trace with an offline-optimal placement.

### 4.2 Reuse per moved byte — the economic objective

A block may cross the boundary once and be read by attention hundreds of times from HBM. That
ratio is the reason residency is worth paying for:

```
                 Σ_b (attention reads of b while resident) · M_b
reuse ratio  =  ────────────────────────────────────────────────
                        Σ_b (n_b + δ_b) · M_b
```

A block that round-trips once (traffic 2·M_b) and is read 500 times during a generation has a
reuse ratio of 250. **Maximising useful HBM reuse per tier byte moved is a better policy
objective than hit rate**, because hit rate is blind to what the hit cost to arrange.

### 4.3 Instrumentation set

| Metric | Instruments |
|---|---|
| HBM byte-time (GiB·s) per completed request | the objective's first term |
| Tier bytes per generated token | the second term; easiest to measure, easiest to compare |
| `A_read`, `A_traffic` vs offline-optimal | placement quality |
| Reuse ratio | whether residency is earning its cost |
| Exposed stall per token | whether HIDING holds |
| Prefetch accuracy (fetched-and-used / fetched) | prefetch policy quality |
| TTFT, inter-token latency, throughput | QoS constraint satisfaction |
| Host CPU utilisation attributable to KV movement | the control/data-plane claim (§2.2) |
| Per-tenant QoS attainment under contention | admission fairness |

All are obtainable on host-DRAM-over-PCIe before any CXL hardware is procured.

---

## 5. What this specification does and does not settle

**Now measured (was assumed):** the decode cost model. §3.4's objective assumed a single
memory bandwidth. Measurement on GB10 shows two, with KV read at ~26% of achievable bandwidth
against 87% for weights — a **~3.4× model-derived effective-path ratio**, with similar KV slopes across two tested frameworks and
two attention kernels. The objective's byte-counting terms should therefore be weighted: a KV
byte is not interchangeable with a weight byte. See `measured_results.md`.

**Settles:** what a KV object is; which metadata migrates and which is updatable in place; where
policy lives versus mechanism; what is being minimised subject to what; how the lossless/lossy
boundary is expressed formally (§3.2); how movement is measured without the accounting error.

**Does not settle:** whether device-side residency of the mechanism is justified (§2.2 lists the
four falsifiable claims); whether persistent policy state changes outcomes materially; what the
workload-specific amplification floors are; whether any of this is novel — that requires a
systematic literature review, which nothing in this chapter constitutes.

**The cost model is now partly measured; the policy claims are not.** §0.1 of the chapter
section gives measured bandwidth terms on one machine. Everything about admission, residency
budgets, metadata persistence and device-side control remains specification, not result. Its value is that the remaining questions are now stated precisely enough to be settled
by experiment rather than argument.
