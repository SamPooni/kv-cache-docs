import numpy as np, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("m", Path(__file__).with_name("kv_tiering_sim_rev1_retained_for_audit.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

shared, blocks, turns = m.build_workload(1)

print("=== alpha sensitivity for EMA (HBM=32 GiB, seed 1) ===")
print(f"{'alpha':>8} {'hit%':>8}")
for a in [0.5, 0.15, 0.05, 0.01, 0.003, 0.001]:
    r = m.run("ema", 32, shared, blocks, turns, alpha=a)
    print(f"{a:>8} {r['hit_rate']:>8.2f}")
for pol in ["lru","lfu"]:
    r = m.run(pol, 32, shared, blocks, turns)
    print(f"{pol:>8} {r['hit_rate']:>8.2f}")

print("\n=== latency decomposition (HBM=32 GiB, seed 1) ===")
for pol in ["lru","lfu","ema"]:
    r = m.run(pol, 32, shared, blocks, turns)
    n = len(turns)
    tot_blocks = r['hit_rate']+r['cxl_rate']+r['cold_rate']
    # reconstruct per-turn ms contributions
    per_turn_blocks = (len(shared)+sum(len(blocks[s]) for s in turns))/n
    cxl_ms  = per_turn_blocks*(r['cxl_rate']/100)*m.CXL_S_PER_BLOCK*1000
    cold_ms = per_turn_blocks*(r['cold_rate']/100)*m.RECOMPUTE_S_PER_BLOCK*1000
    print(f"{pol:>5}  hit {r['hit_rate']:5.1f}%  cxl-fetch {cxl_ms:7.2f} ms  "
          f"recompute {cold_ms:7.2f} ms  ({100*cold_ms/(cxl_ms+cold_ms):.1f}% of stall)")
