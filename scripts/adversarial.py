import numpy as np, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("m", Path(__file__).with_name("kv_tiering_sim_rev1_retained_for_audit.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
GIB, BPB = m.GIB, m.BYTES_PER_BLOCK

def synth(seed, mode, n_sessions=200, n_turns=2500):
    rng = np.random.default_rng(seed)
    S = 100_000
    shared = list(range(128))
    ctx = rng.integers(1024, 12288, size=n_sessions)
    blocks = {s: [(s+1)*S+i for i in range(int(ctx[s])//m.BLOCK_TOKENS)] for s in range(n_sessions)}
    if mode == "scan":          # tiny hot set + flood of one-shot sessions
        turns=[]
        for i in range(n_turns):
            turns.append(int(rng.integers(0,5)) if i%2==0 else int(rng.integers(5,n_sessions)))
    elif mode == "loop":        # cyclic sweep slightly larger than cache -- LRU's worst case
        turns=[i % 40 for i in range(n_turns)]
    return shared, blocks, turns

for mode in ["scan","loop"]:
    print(f"=== {mode} workload, HBM=32 GiB ===")
    for pol,kw in [("lru",{}),("lfu",{}),("ema",{"alpha":0.15}),("ema",{"alpha":0.01})]:
        hs=[]
        for seed in [1,2,3]:
            sh,bl,tu = synth(seed,mode)
            hs.append(m.run(pol,32,sh,bl,tu,**kw)["hit_rate"])
        tag = f"{pol}" + (f"(a={kw['alpha']})" if kw else "")
        print(f"  {tag:>12} {np.mean(hs):6.2f}% +- {np.std(hs):.2f}")
    print()
