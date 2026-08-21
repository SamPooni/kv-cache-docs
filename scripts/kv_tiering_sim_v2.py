"""
KV-cache tiering simulation v2 — HBM <-> CXL <-> recompute.

Changes from v1, all in response to review:

  [FIX-1] Eviction is now EXACT for every policy. v1's lazy heap re-pushed only when
          the recomputed score EXCEEDED the stored key. That is correct for LFU (counts
          rise, keys are stale-low) but wrong for a decaying EMA (scores fall, keys are
          stale-high), where it selected the wrong victim on 99.6% of evictions.
          Replaced with exact order-preserving keys plus a version check:

            EMA:  S(t) = S_touch · (1-a)^(t - t_touch)
                  ln S(t) = [ln S_touch - t_touch·ln(1-a)] + t·ln(1-a)
                  The bracketed term is time-invariant and the trailing term is common
                  to all blocks at any fixed t, so ordering by
                      K = ln S_touch + t_touch·(-ln(1-a))
                  is EXACTLY ordering by current score. No re-evaluation needed.
                  Range check: t <= 1e4, -ln(0.85) = 0.1625 -> K < 1700. No overflow.

            LFU:  K = access count, likewise exact.

          Stale heap entries are skipped by comparing against self.key[b].

  [FIX-2] Cold-start and warm steady-state are now separated. v1 averaged compulsory
          first-touch recompute over the whole run, which swamped policy differences.
          `warmup_turns` excludes the population phase from all statistics.

  [FIX-3] Policy metadata optionally survives HBM eviction (`persist_meta`), so
          "HBM-residency-local frequency" and "global block frequency" can be compared
          as distinct policies rather than conflated.

  [FIX-4] Decode length is modelled. Each turn generates D tokens; blocks are appended
          as generation proceeds, so a 1-token reply and a 500-token reply no longer
          look identical.

  [FIX-5] Optional idealised prefetch/compute overlap (`overlap`): CXL transfer for a
          turn is hidden behind decode compute up to the available decode time.

Modelling scope, stated plainly: this is a block-level trace simulation approximating
multi-turn serving behaviour. It does not model HBM bandwidth, memory-controller
contention, transfer queueing, attention kernel access order, CUDA graph reservations,
allocator fragmentation, batch composition, or tensor-parallel sharding.
"""

import heapq
import math
import numpy as np
from collections import OrderedDict, defaultdict

# ---------------------------------------------------------------- model geometry
LAYERS, KV_HEADS, HEAD_DIM, DTYPE_B = 80, 8, 128, 2      # Llama-3 70B, bf16
BLOCK_TOKENS = 16
BYTES_PER_TOKEN = 2 * LAYERS * KV_HEADS * HEAD_DIM * DTYPE_B
BYTES_PER_BLOCK = BYTES_PER_TOKEN * BLOCK_TOKENS
GIB = 1024 ** 3

# ---------------------------------------------------------------- cost model
CXL_BW_BPS = 64e9
CXL_LAT_S = 300e-9
PARAMS = 70e9
EFF_FLOPS = 125e12
# First-order LOWER BOUND on regenerating one block: the linear prefill term only.
# Regenerating an evicted interior block in a real engine needs the preceding context's
# hidden states, so practical cost is higher -- see report section 4/appendix B.
RECOMPUTE_S_PER_BLOCK = (2 * PARAMS * BLOCK_TOKENS) / EFF_FLOPS
CXL_S_PER_BLOCK = BYTES_PER_BLOCK / CXL_BW_BPS + CXL_LAT_S
# Decode is memory-bandwidth bound; ~1 forward pass per token over the weights.
HBM_BW_BPS = 3.35e12                      # H100 SXM class, per GPU
TP_DEGREE = 4                             # 4-way tensor parallel; weights sharded
# Theoretical floor: each GPU streams its shard of the weights once per token.
# Real engines reach roughly 40% of this; treated as a floor, used only for overlap.
DECODE_S_PER_TOKEN = (PARAMS * 2 / TP_DEGREE) / HBM_BW_BPS


class Tier:
    """Capacity-bounded block store with an exact pluggable eviction policy."""

    def __init__(self, capacity_blocks, policy, alpha=0.15, persist_meta=False):
        self.cap = capacity_blocks
        self.policy = policy
        self.alpha = alpha
        self.persist_meta = persist_meta
        self.neg_log = -math.log(1.0 - alpha) if 0.0 < alpha < 1.0 else 0.0

        self.resident = set()
        self.lru = OrderedDict()
        self.s_touch = {}        # EMA score value at last touch
        self.t_touch = {}        # time of last touch
        self.count = {}          # LFU
        self.key = {}            # current authoritative heap key per block
        self.heap = []
        self.evicted = []

    # -- exact keys ---------------------------------------------------------
    def _cur_score(self, b, t):
        """True current score. Used for auditing only; eviction does not need it."""
        if self.policy == "lfu":
            return self.count[b]
        return self.s_touch[b] * (1.0 - self.alpha) ** (t - self.t_touch[b])

    def _make_key(self, b, t):
        if self.policy == "lfu":
            return float(self.count[b])
        # order-preserving transform of the decayed EMA score
        return math.log(self.s_touch[b]) + self.t_touch[b] * self.neg_log

    def touch(self, b, t, weight=1.0):
        if self.policy == "lru":
            self.lru[b] = t
            self.lru.move_to_end(b)
            return
        if self.policy == "lfu":
            self.count[b] = self.count.get(b, 0.0) + weight
        else:
            if b in self.s_touch:
                prev = self.s_touch[b] * (1.0 - self.alpha) ** (t - self.t_touch[b])
            else:
                prev = 0.0
            s = self.alpha * weight + (1.0 - self.alpha) * prev
            self.s_touch[b] = s if s > 0.0 else self.alpha * weight
            self.t_touch[b] = t
        k = self._make_key(b, t)
        self.key[b] = k
        heapq.heappush(self.heap, (k, b))

    # -- membership ---------------------------------------------------------
    def admit(self, b, t, weight=1.0):
        self.resident.add(b)
        self.touch(b, t, weight)
        self._enforce(t)

    def _enforce(self, t):
        while len(self.resident) > self.cap:
            victim = self._pick_victim(t)
            if victim is None:
                break
            self.resident.discard(victim)
            if not self.persist_meta:
                self.s_touch.pop(victim, None)
                self.t_touch.pop(victim, None)
                self.count.pop(victim, None)
            self.key.pop(victim, None)
            self.lru.pop(victim, None)
            self.evicted.append(victim)

    def _pick_victim(self, t):
        if self.policy == "lru":
            return next(iter(self.lru)) if self.lru else None
        while self.heap:
            k, b = heapq.heappop(self.heap)
            if b not in self.resident:
                continue
            cur = self.key.get(b)
            if cur is None or cur != k:      # stale entry from an earlier touch
                continue
            return b
        return next(iter(self.resident)) if self.resident else None


# ---------------------------------------------------------------- workload
def build_workload(seed, n_sessions=200, n_turns=2500, shared_prompt_tokens=2048,
                   mode="zipf", decode_lo=32, decode_hi=512):
    """Multi-turn traffic. Returns (shared_blocks, session_blocks, turns, decode_lens).

    `turns` is a list of session ids; `decode_lens` the generated-token count per turn.
    Session context grows as turns are served (see grow()).
    """
    rng = np.random.default_rng(seed)
    STRIDE = 1_000_000
    shared = list(range(shared_prompt_tokens // BLOCK_TOKENS))

    ctx = rng.integers(1024, 12288, size=n_sessions)
    blocks = {s: [(s + 1) * STRIDE + i for i in range(int(ctx[s]) // BLOCK_TOKENS)]
              for s in range(n_sessions)}

    if mode == "zipf":
        p = 1.0 / np.arange(1, n_sessions + 1) ** 1.1
        p /= p.sum()
        order = rng.permutation(n_sessions)
        turns = [int(order[i]) for i in rng.choice(n_sessions, size=n_turns, p=p)]
    elif mode == "scan":
        turns = [int(rng.integers(0, 5)) if i % 2 == 0 else int(rng.integers(5, n_sessions))
                 for i in range(n_turns)]
    elif mode == "loop":
        turns = [i % 40 for i in range(n_turns)]
    else:
        raise ValueError(mode)

    decode = rng.integers(decode_lo, decode_hi, size=n_turns)
    return shared, blocks, turns, [int(d) for d in decode]


def run(policy, hbm_gib, wl, cxl_gib=512, alpha=0.15, persist_meta=False,
        warmup_turns=0, overlap=False, grow=True):
    """Simulate. Statistics are collected only for turns >= warmup_turns."""
    shared, blocks, turns, decode = wl
    blocks = {s: list(v) for s, v in blocks.items()}          # local copy; may grow
    next_idx = {s: len(v) for s, v in blocks.items()}
    STRIDE = 1_000_000

    hbm = Tier(int(hbm_gib * GIB // BYTES_PER_BLOCK), policy, alpha, persist_meta)
    cxl = Tier(int(cxl_gib * GIB // BYTES_PER_BLOCK), "lru")

    hits = m_cxl = m_cold = 0
    stall = 0.0
    fetch_s = 0.0
    recomp_s = 0.0
    counted_turns = 0

    for t, s in enumerate(turns):
        live = t >= warmup_turns
        ws = shared + blocks[s]
        need_fetch, need_recompute = [], []

        for b in ws:
            if b in hbm.resident:
                if live:
                    hits += 1
                hbm.touch(b, t)
            elif b in cxl.resident:
                if live:
                    m_cxl += 1
                need_fetch.append(b)
            else:
                if live:
                    m_cold += 1
                need_recompute.append(b)

        for b in need_fetch + need_recompute:
            hbm.admit(b, t)
        for b in hbm.evicted:
            cxl.admit(b, t)
        hbm.evicted.clear()
        cxl.evicted.clear()

        f = len(need_fetch) * CXL_S_PER_BLOCK
        r = len(need_recompute) * RECOMPUTE_S_PER_BLOCK
        if overlap:
            # idealised: transfer hides behind this turn's decode compute
            f = max(0.0, f - decode[t] * DECODE_S_PER_TOKEN)
        if live:
            fetch_s += f
            recomp_s += r
            stall += f + r
            counted_turns += 1

        # context grows with the generated tokens
        if grow:
            new = decode[t] // BLOCK_TOKENS
            for _ in range(int(new)):
                nb = (s + 1) * STRIDE + next_idx[s]
                next_idx[s] += 1
                blocks[s].append(nb)
                hbm.admit(nb, t)
            for b in hbm.evicted:
                cxl.admit(b, t)
            hbm.evicted.clear()
            cxl.evicted.clear()

    total = hits + m_cxl + m_cold
    n = max(counted_turns, 1)
    return {
        "policy": policy, "alpha": alpha, "hbm_gib": hbm_gib, "cxl_gib": cxl_gib,
        "persist_meta": persist_meta, "overlap": overlap,
        "hit_rate": 100.0 * hits / max(total, 1),
        "cxl_rate": 100.0 * m_cxl / max(total, 1),
        "cold_rate": 100.0 * m_cold / max(total, 1),
        "stall_ms": 1000.0 * stall / n,
        "fetch_ms": 1000.0 * fetch_s / n,
        "recompute_ms": 1000.0 * recomp_s / n,
        "turns": n,
    }


if __name__ == "__main__":
    print(f"bytes/token       : {BYTES_PER_TOKEN/1024:.1f} KiB")
    print(f"bytes/block (16)  : {BYTES_PER_BLOCK/1024/1024:.2f} MiB")
    print(f"CXL fetch/block   : {CXL_S_PER_BLOCK*1e6:.1f} us")
    print(f"recompute/block   : {RECOMPUTE_S_PER_BLOCK*1e3:.2f} ms (lower bound)")
    print(f"decode/token      : {DECODE_S_PER_TOKEN*1e3:.2f} ms\n")

    wl = build_workload(1)
    for pol, kw in [("lru", {}), ("lfu", {}), ("ema", {"alpha": 0.15}),
                    ("ema", {"alpha": 0.01})]:
        cold = run(pol, 32, wl, **kw)
        warm = run(pol, 32, wl, warmup_turns=1250, **kw)
        tag = pol + (f" a={kw['alpha']}" if kw else "")
        print(f"{tag:>11}  cold hit {cold['hit_rate']:5.2f}%  stall {cold['stall_ms']:7.1f} ms"
              f"   |  warm hit {warm['hit_rate']:5.2f}%  stall {warm['stall_ms']:7.1f} ms"
              f"  (recompute {100*warm['recompute_ms']/max(warm['stall_ms'],1e-9):4.1f}%)")
