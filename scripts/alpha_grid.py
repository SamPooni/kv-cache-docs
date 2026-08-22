import json, importlib.util, numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
def load():
    s=importlib.util.spec_from_file_location("m", Path(__file__).with_name("kv_tiering_sim_v2.py"))
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def job(a):
    alpha,hb,sd,pm=a; m=load(); wl=m.build_workload(sd)
    r=m.run("ema",hb,wl,warmup_turns=1250,alpha=alpha,persist_meta=pm)
    return (alpha,hb,sd,pm),r["hit_rate"]
jobs=[(al,hb,sd,pm) for al in [0.3,0.15,0.08,0.05,0.03,0.01,0.003]
      for hb in [16,32,64] for sd in (1,2) for pm in (False,True)]
print("jobs",len(jobs),flush=True)
out={}
with ProcessPoolExecutor(max_workers=4) as ex:
    for i,(k,v) in enumerate(ex.map(job,jobs),1):
        out["|".join(map(str,k))]=v
        if i%24==0: print(f" {i}/{len(jobs)}",flush=True)
json.dump(out,open(Path(__file__).parents[1]/"data"/"alpha_grid.json","w"),indent=1)
print("done")
