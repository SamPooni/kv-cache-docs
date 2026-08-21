import json, numpy as np
R=json.load(open("results_v2.json"))
POL=["LRU","LFU","EMA a=0.15","EMA a=0.01"]
g=lambda k: R[k]

print("=== E1  WARM steady-state by HBM budget (2 seeds) ===")
print(f"{'HBM':>5} " + "".join(f"{p:>20}" for p in POL))
e1={}
for hb in [8,16,32,64,128]:
    cells=[]
    for p in POL:
        h=[g(f"e1|{hb}|{p}|{s}")["hit_rate"] for s in (1,2)]
        st=[g(f"e1|{hb}|{p}|{s}")["stall_ms"] for s in (1,2)]
        e1[(hb,p)]=(np.mean(h),np.std(h),np.mean(st))
        cells.append(f"{np.mean(h):>12.2f}+-{np.std(h):<5.2f}")
    print(f"{hb:>4}G "+"".join(cells))
print("\n  stall ms/turn")
for hb in [8,16,32,64,128]:
    print(f"{hb:>4}G "+"".join(f"{e1[(hb,p)][2]:>20.1f}" for p in POL))

print("\n=== E2  alpha sweep (32 GiB, warm) ===")
for a in [0.5,0.3,0.15,0.05,0.01,0.003,0.001]:
    print(f"  a={a:<6} {g(f'e2|{a}')['hit_rate']:6.2f}%")
for n in ["LRU","LFU"]:
    print(f"  {n:<8} {g(f'e2|{n}')['hit_rate']:6.2f}%")

print("\n=== E3  workloads (32 GiB, warm, 3 seeds) ===")
print(f"{'workload':>8} " + "".join(f"{p:>20}" for p in POL))
for mode in ["zipf","scan","loop"]:
    cells=[]
    for p in POL:
        h=[g(f"e3|{mode}|{p}|{s}")["hit_rate"] for s in (1,2,3)]
        cells.append(f"{np.mean(h):>12.2f}+-{np.std(h):<5.2f}")
    print(f"{mode:>8} "+"".join(cells))

print("\n=== E4  cold-start vs warm (32 GiB) ===")
print(f"{'policy':>11} {'cold hit':>9} {'cold stall':>11} {'cold rec%':>10} "
      f"{'warm hit':>9} {'warm stall':>11} {'warm rec%':>10}")
for p in POL:
    c=g(f"e4cold|{p}"); w=g(f"e4warm|{p}")
    cr=100*c["recompute_ms"]/max(c["stall_ms"],1e-9)
    wr=100*w["recompute_ms"]/max(w["stall_ms"],1e-9)
    print(f"{p:>11} {c['hit_rate']:>8.2f}% {c['stall_ms']:>10.1f} {cr:>9.1f}% "
          f"{w['hit_rate']:>8.2f}% {w['stall_ms']:>10.1f} {wr:>9.1f}%")

print("\n=== E5  metadata persistence across HBM eviction (32 GiB, warm) ===")
for p in POL[1:]:
    l=g(f"e5loc|{p}")["hit_rate"]; q=g(f"e5per|{p}")["hit_rate"]
    print(f"  {p:>11}  local {l:6.2f}%   persistent {q:6.2f}%   delta {q-l:+6.2f}")

print("\n=== E6  idealised transfer/compute overlap (32 GiB, warm) ===")
for p in POL:
    n=g(f"e6no|{p}"); o=g(f"e6ov|{p}")
    print(f"  {p:>11}  stall {n['stall_ms']:7.1f} -> {o['stall_ms']:7.1f} ms   "
          f"fetch {n['fetch_ms']:6.2f} -> {o['fetch_ms']:5.2f} ms")
