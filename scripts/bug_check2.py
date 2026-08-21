"""Re-audit victim selection against brute-force argmin, on the v2 Tier."""
import importlib.util
spec=importlib.util.spec_from_file_location("m2","kv_tiering_sim_v2.py")
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
            self.checks+=1; vs=self._cur_score(v,t)
            if vs>ts+1e-12:
                self.wrong+=1; self.worst=max(self.worst,vs-ts)
        return v

sh,bl,tu,dec=m.build_workload(1,n_sessions=60,n_turns=250)
for pol,a in [("ema",0.15),("ema",0.01),("ema",0.5),("lfu",0.15)]:
    hbm=Audited(int(8*m.GIB//m.BYTES_PER_BLOCK),pol,a)
    for t,s in enumerate(tu):
        for b in sh+bl[s]:
            if b in hbm.resident: hbm.touch(b,t)
            else: hbm.admit(b,t)
        hbm.evicted.clear()
    tag=f"{pol} a={a}"
    print(f"{tag:>12}: {hbm.wrong:>6}/{hbm.checks:<6} wrong ({100*hbm.wrong/max(hbm.checks,1):5.2f}%)  worst err {hbm.worst:.3e}")
