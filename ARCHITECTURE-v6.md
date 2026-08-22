# The KV-State Control Plane — Architecture Reference (v6.0)

Sam Pooni

---

This document is the formal, standalone reference for v6.0's architecture.
**Chapter 3 of the main package ("The KV-State Control Plane") is the
narrative version of the same content** — read this document when you want
the structure on its own, without the surrounding prose; read Chapter 3
when you want it in context with the rest of the package's argument.

The architecture below is a **proposed design**, not a validated system.
Architecture is a design artifact; measurement validates performance
claims, not whether something qualifies as an architecture. Only one piece
of it — residency persistence inside PLACE — carries empirical evidence
today, and that evidence is simulated and provisional. See the evidence
table at the end of this document for the complete accounting.

## 1. Two orthogonal hierarchies

The architecture is organized around two hierarchies that ask different
questions and do not collapse into one another.

### 1.1 Information hierarchy — "what is worth considering?"

| Level | Description |
|---|---|
| Agent/model state | Task, conversation, and tool-call context driving the current step |
| Context-region summaries | Coarse summaries over spans of context (e.g., a tool-call turn, a document section) |
| Candidate regions | Narrowed-down spans identified as plausibly relevant |
| KV blocks | Fixed-size groups of token-level KV (this package's existing unit, e.g. 16-token blocks) |
| Token-level KV | Individual key/value pairs, the finest grain |

```
Agent/Model State
      │
      ▼
Context-Region Summaries
      │
      ▼
Candidate Regions
      │
      ▼
KV Blocks
      │
      ▼
Token-Level KV
```

### 1.2 Physical hierarchy — "where should it live?"

| Tier | Description |
|---|---|
| HBM | GPU-local high-bandwidth memory — fastest, smallest, most expensive |
| Host / CXL DRAM | Host-attached or CXL-pooled DRAM — larger, higher latency than HBM |
| NVMe / remote storage | Largest, cheapest, highest latency |

```
HBM
 │
 ▼
Host / CXL DRAM
 │
 ▼
NVMe / Remote Storage
```

The information hierarchy decides *what* matters; the physical hierarchy
decides *where it can live*. The control plane below is the thing that
maps decisions in the first hierarchy onto placement and movement in the
second.

## 2. The control-plane pipeline

```
MODEL / AGENT STATE
        │
        ├── relevance signals (information hierarchy)
        └── reuse / deadline signals (tool state, identity, timing)
        │
        ▼
┌───────────────────────────────┐
│   KV-STATE CONTROL PLANE       │
│                                 │
│   SELECT  →  PLACE  →  MOVE    │
└───────────────────────────────┘
        │
        ▼
     ATTEND
        │
        ▼
     OBSERVE
        │
        └──── feedback ────► back to MODEL / AGENT STATE
```

- **SELECT** — decide which candidate regions / KV blocks are worth
  considering at all, using relevance and reuse-prediction signals.
- **PLACE** — decide which physical tier a selected block should occupy,
  including whether previously-computed policy/residency state for that
  block should carry over across a tier boundary. (This is where v5's
  persistent-metadata mechanism lives in v6 — see §4.)
- **MOVE** — execute the physical transfer between tiers, including
  scheduling, batching, and hiding movement cost behind compute.
- **ATTEND** — the model consumes the placed/selected KV state.
- **OBSERVE** — record what was actually used/attended, closing the loop.
- **Feedback** — OBSERVE's output updates agent/model state and signal
  sources feeding the next SELECT decision.

## 3. Past-only vs. predictive signals

SELECT (and, to a lesser extent, PLACE) can be driven by two different
classes of signal.

| Past-only signals | Predictive signals |
|---|---|
| Recency (most-recently-used) | Agent/tool-call state (what the agent is currently doing) |
| Frequency (access counts) | Tool-return timing (when a result is expected back) |
| EMA / LRFU-decayed scores | Prefix identity (recognizing a recurring prompt/context prefix) |
| | Reuse history (has this exact region been reused before, and how often) |
| | Attention signals (what the model actually attended to recently) |

Past-only signals describe what has already happened to a block (LRU, LFU,
EMA-decayed variants — the policy classes evaluated in this package's own
simulation). Predictive signals attempt to anticipate what will be needed
next, using information outside the raw access history — this is the
category CacheWise (tool-call-driven reuse prediction) and PNM-KV
(near-memory selection before movement) occupy in Chapter 10's landscape.
v6's SELECT stage is designed to accommodate both; only past-only signals
have been evaluated by this package's own simulation to date.

## 4. Evidence table

| Architectural piece | Status | Basis |
|---|---|---|
| DGX Spark GB10 decode-bandwidth decomposition (236.5 GB/s achieved, ~3.4× KV-vs-weight byte cost) | **Measured** | Hardware measurement, DGX Spark GB10 |
| PLACE — residency-state persistence across a tier boundary (+34.15 pts, LFU @ 32 GiB HBM) | **Simulated — provisional** | `kv_tiering_sim_v2.py`, Revision 2 |
| SELECT — relevance estimation via predictive signals | **Design / unvalidated** | Not implemented or measured by this package |
| SELECT — reuse prediction from agent/tool-call state | **Design / unvalidated** | Adjacent published work (CacheWise) exists; not reproduced here |
| MOVE — cost and hiding/overlap on real CXL hardware | **Design / unvalidated** | No CXL device has been measured by this package |
| Information-hierarchy representation (context-region summaries, candidate regions) | **Design / unvalidated** | No representation has been chosen or tested |
| Mechanism placement (which technique belongs in which stage) | **Design / unvalidated** | Open question; see Chapter 10 §10.3 |
| OBSERVE → feedback loop | **Design / unvalidated** | Conceptual only |
| 6× memory expansion, 320 KiB/token, 5.00 MiB blocks, capacity table | **Analytical** | Arithmetic over stated capacities; independent of the above |

Nothing in this document should be read as claiming SELECT, MOVE, ATTEND,
OBSERVE, or the feedback loop have been built or tested — only PLACE's
residency-persistence mechanism has any empirical result behind it, and
that result is simulated, not measured on real hardware.

---

© 2025–2026 Subramaniyam (Sam) Pooni. All Rights Reserved.
