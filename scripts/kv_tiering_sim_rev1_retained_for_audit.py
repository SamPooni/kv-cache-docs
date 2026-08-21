"""
KV-cache tiering simulation: HBM <-> CXL <-> recompute, under a multi-turn agentic workload.

What this models (and the previous draft did not):
  * A real caching UNIT. KV cache is keyed by (session, block-of-16-tokens), not by
    attention head. You cannot evict "head 7" and still compute attention correctly --
    every head of every layer is required for every token. Head-granular work in the
    literature (H2O, SnapKV) evicts (head, TOKEN) pairs; the token axis is not optional.
  * Real bytes. Llama-3 70B GQA: 8 KV heads, not 64.
  * A real access pattern. Attention at turn N reads the ENTIRE prior context, so a
    session's whole block set must be resident for its turn.
  * An actual baseline. LRU and LFU are implemented and run, not described.
  * A cost model. Misses cost bytes/bandwidth; cold blocks cost a prefill recompute.
"""

import heapq
import numpy as np
from collections import OrderedDict, defaultdict

# ---------------------------------------------------------------- model geometry
LAYERS, KV_HEADS, HEAD_DIM, DTYPE_B = 80, 8, 128, 2      # Llama-3 70B, bf16
BLOCK_TOKENS = 16
BYTES_PER_TOKEN = 2 * LAYERS * KV_HEADS * HEAD_DIM * DTYPE_B      # K and V
BYTES_PER_BLOCK = BYTES_PER_TOKEN * BLOCK_TOKENS
GIB = 1024 ** 3

# ---------------------------------------------------------------- cost model
CXL_BW_BPS = 64e9          # CXL 3.0 x16, usable one direction
CXL_LAT_S = 300e-9         # per-transfer setup
PARAMS = 70e9
EFF_FLOPS = 125e12         # H100-class, ~40% MFU on prefill
RECOMPUTE_S_PER_BLOCK = (2 * PARAMS * BLOCK_TOKENS) / EFF_FLOPS
CXL_S_PER_BLOCK = BYTES_PER_BLOCK / CXL_BW_BPS + CXL_LAT_S


class Tier:
    """Capacity-bounded block store with a pluggable eviction policy."""

    def __init__(self, capacity_blocks, policy, alpha=0.15):
        self.cap = capacity_blocks
        self.policy = policy
        self.alpha = alpha
        self.resident = set()
        self.lru = OrderedDict()                 # policy == "lru"
        self.score = {}                          # policy in ("lfu", "ema")
        self.last_t = {}
        self.heap = []
        self.evicted = []                        # drained by caller each step

    # -- scoring ------------------------------------------------------------
    def _cur_score(self, b, t):
        if self.policy == "lfu":
            return self.score[b]
        # EMA with event-driven decay == LRFU (Lee et al., 1999). Not a new algorithm.
        return self.score[b] * (1.0 - self.alpha) ** (t - self.last_t[b])

    def touch(self, b, t, weight=1.0):
        if self.policy == "lru":
            self.lru[b] = t
            self.lru.move_to_end(b)
            return
        if self.policy == "lfu":
            self.score[b] = self.score.get(b, 0.0) + weight
        else:
            prev = self._cur_score(b, t) if b in self.score else 0.0
            self.score[b] = self.alpha * weight + (1.0 - self.alpha) * prev
            self.last_t[b] = t
        heapq.heappush(self.heap, (self.score[b], b))

    # -- membership ---------------------------------------------------------
    def admit(self, b, t, weight=1.0):
        if b not in self.resident:
            self.resident.add(b)
        self.touch(b, t, weight)
        self._enforce(t)

    def _enforce(self, t):
        while len(self.resident) > self.cap:
            victim = self._pick_victim(t)
            if victim is None:
                break
            self.resident.discard(victim)
            self.score.pop(victim, None)
            self.last_t.pop(victim, None)
            self.lru.pop(victim, None)
            self.evicted.append(victim)

    def _pick_victim(self, t):
        if self.policy == "lru":
            return next(iter(self.lru)) if self.lru else None
        # lazy heap: keys are stale-high, so re-push the recomputed value until stable
        while self.heap:
            k, b = heapq.heappop(self.heap)
            if b not in self.resident:
                continue
            cur = self._cur_score(b, t)
            if cur > k + 1e-12:
                heapq.heappush(self.heap, (cur, b))
                continue
            return b
        return next(iter(self.resident)) if self.resident else None


def build_workload(seed, n_sessions=200, n_turns=2500, shared_prompt_tokens=2048):
    """Multi-turn agentic traffic: a shared system/RAG prefix plus per-session context.

    Session popularity is Zipf -- a few conversations come back constantly, most are
    near one-shot. This is the pattern that makes a scoring policy plausibly useful.
    """
    rng = np.random.default_rng(seed)
    # integer block ids so heap tie-breaks never compare mixed types
    STRIDE = 100_000
    shared = [i for i in range(shared_prompt_tokens // BLOCK_TOKENS)]

    ctx_tokens = rng.integers(1024, 12288, size=n_sessions)
    blocks = {s: [(s + 1) * STRIDE + i for i in range(int(ctx_tokens[s]) // BLOCK_TOKENS)]
              for s in range(n_sessions)}

    ranks = np.arange(1, n_sessions + 1)
    p = 1.0 / ranks ** 1.1
    p /= p.sum()
    order = rng.permutation(n_sessions)
    turns = [int(order[i]) for i in rng.choice(n_sessions, size=n_turns, p=p)]
    return shared, blocks, turns


def run(policy, hbm_gib, shared, blocks, turns, cxl_gib=512, alpha=0.15):
    hbm = Tier(int(hbm_gib * GIB // BYTES_PER_BLOCK), policy, alpha)
    cxl = Tier(int(cxl_gib * GIB // BYTES_PER_BLOCK), "lru")

    hits = misses_cxl = misses_cold = 0
    latency = 0.0

    for t, s in enumerate(turns):
        working_set = shared + blocks[s]
        need_cxl, need_recompute = [], []

        for b in working_set:
            if b in hbm.resident:
                hits += 1
                hbm.touch(b, t)
            elif b in cxl.resident:
                misses_cxl += 1
                need_cxl.append(b)
            else:
                misses_cold += 1
                need_recompute.append(b)

        # fetched and recomputed blocks are promoted into HBM
        for b in need_cxl + need_recompute:
            hbm.admit(b, t)
        for b in hbm.evicted:
            cxl.admit(b, t)          # spill, not drop
        hbm.evicted.clear()
        cxl.evicted.clear()

        latency += len(need_cxl) * CXL_S_PER_BLOCK
        latency += len(need_recompute) * RECOMPUTE_S_PER_BLOCK

    total = hits + misses_cxl + misses_cold
    return {
        "policy": policy,
        "hbm_gib": hbm_gib,
        "hit_rate": 100.0 * hits / total,
        "cxl_rate": 100.0 * misses_cxl / total,
        "cold_rate": 100.0 * misses_cold / total,
        "mean_turn_stall_ms": 1000.0 * latency / len(turns),
    }


if __name__ == "__main__":
    print(f"bytes/token      : {BYTES_PER_TOKEN/1024:.1f} KiB")
    print(f"bytes/block (16) : {BYTES_PER_BLOCK/1024/1024:.2f} MiB")
    print(f"8K context       : {BYTES_PER_TOKEN*8192/GIB:.2f} GiB")
    print(f"CXL fetch/block  : {CXL_S_PER_BLOCK*1e6:.1f} us")
    print(f"recompute/block  : {RECOMPUTE_S_PER_BLOCK*1e3:.2f} ms "
          f"({RECOMPUTE_S_PER_BLOCK/CXL_S_PER_BLOCK:.0f}x the fetch)\n")

    budgets = [8, 16, 32, 64, 128]
    policies = ["lru", "lfu", "ema"]
    seeds = [1, 2, 3]
    agg = defaultdict(list)

    for seed in seeds:
        shared, blocks, turns = build_workload(seed)
        footprint = (len(shared) + sum(len(v) for v in blocks.values())) * BYTES_PER_BLOCK
        if seed == seeds[0]:
            print(f"total KV footprint: {footprint/GIB:.0f} GiB across 200 sessions\n")
        for hb in budgets:
            for pol in policies:
                r = run(pol, hb, shared, blocks, turns)
                agg[(hb, pol)].append(r)

    print(f"{'HBM':>5} {'policy':>7} {'hit%':>14} {'cold%':>7} {'stall ms/turn':>16}")
    for hb in budgets:
        for pol in policies:
            rs = agg[(hb, pol)]
            h = np.array([r["hit_rate"] for r in rs])
            c = np.array([r["cold_rate"] for r in rs])
            m = np.array([r["mean_turn_stall_ms"] for r in rs])
            print(f"{hb:>4}G {pol:>7} {h.mean():>8.2f}+-{h.std():<4.2f} "
                  f"{c.mean():>6.2f} {m.mean():>11.1f}+-{m.std():<4.1f}")
        print()

    np.save("/home/claude/kv/results.npy",
            np.array([(hb, pol, np.mean([r["hit_rate"] for r in agg[(hb, pol)]]),
                       np.mean([r["mean_turn_stall_ms"] for r in agg[(hb, pol)]]))
                      for hb in budgets for pol in policies], dtype=object),
            allow_pickle=True)
