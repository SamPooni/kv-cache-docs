"""Does _pick_victim actually return the minimum-scoring resident block under EMA decay?

Brute-force the true argmin at every eviction and compare.
"""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location("m", Path(__file__).with_name("kv_tiering_sim_rev1_retained_for_audit.py"))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class Audited(m.Tier):
    def __init__(self,*a,**k):
        super().__init__(*a,**k); self.checks=0; self.wrong=0; self.worst=0.0
    def _pick_victim(self,t):
        truth=None
        if self.policy!="lru" and self.resident:
            truth=min(self.resident,key=lambda b:self._cur_score(b,t))
            ts=self._cur_score(truth,t)
        v=super()._pick_victim(t)
        if truth is not None and v is not None:
            self.checks+=1
            vs=self._cur_score(v,t)
            if vs>ts+1e-12:
                self.wrong+=1
                self.worst=max(self.worst,vs-ts)
        return v

m.Tier_orig=m.Tier
sh,bl,tu=m.build_workload(1,n_sessions=60,n_turns=250)

for pol,alpha in [("ema",0.15),("ema",0.01),("lfu",0.0)]:
    hbm=Audited(int(8*m.GIB//m.BYTES_PER_BLOCK),pol,alpha if alpha else 0.15)
    cxl=m.Tier(int(512*m.GIB//m.BYTES_PER_BLOCK),"lru")
    for t,s in enumerate(tu):
        for b in sh+bl[s]:
            if b in hbm.resident: hbm.touch(b,t)
            else: hbm.admit(b,t)
        hbm.evicted.clear()
    tag=f"{pol} a={alpha}" if pol=="ema" else pol
    pct=100*hbm.wrong/max(hbm.checks,1)
    print(f"{tag:>12}: {hbm.wrong:>6}/{hbm.checks:<6} wrong victims ({pct:5.1f}%)  "
          f"worst score error {hbm.worst:.4f}")
