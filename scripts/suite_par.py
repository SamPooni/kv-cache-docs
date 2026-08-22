import json, numpy as np, importlib.util, itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def load():
    spec=importlib.util.spec_from_file_location("m", Path(__file__).with_name("kv_tiering_sim_v2.py"))
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def job(args):
    (tag,seed,mode,pol,hb,kw,extra)=args
    m=load(); wl=m.build_workload(seed,mode=mode)
    r=m.run(pol,hb,wl,**kw,**extra)
    return tag,r

CFG=[("LRU","lru",{}),("LFU","lfu",{}),("EMA a=0.15","ema",{"alpha":0.15}),
     ("EMA a=0.01","ema",{"alpha":0.01})]
WARM={"warmup_turns":1250}
jobs=[]
for hb in [8,16,32,64,128]:
    for n,p,kw in CFG:
        for sd in (1,2):
            jobs.append((f"e1|{hb}|{n}|{sd}",sd,"zipf",p,hb,kw,WARM))
for a in [0.5,0.3,0.15,0.05,0.01,0.003,0.001]:
    jobs.append((f"e2|{a}",1,"zipf","ema",32,{"alpha":a},WARM))
for n,p,kw in [("LRU","lru",{}),("LFU","lfu",{})]:
    jobs.append((f"e2|{n}",1,"zipf",p,32,kw,WARM))
for mode in ["zipf","scan","loop"]:
    for n,p,kw in CFG:
        for sd in (1,2,3):
            jobs.append((f"e3|{mode}|{n}|{sd}",sd,mode,p,32,kw,WARM))
for n,p,kw in CFG:
    jobs.append((f"e4cold|{n}",1,"zipf",p,32,kw,{}))
    jobs.append((f"e4warm|{n}",1,"zipf",p,32,kw,WARM))
for n,p,kw in CFG[1:]:
    jobs.append((f"e5loc|{n}",1,"zipf",p,32,kw,dict(WARM,persist_meta=False)))
    jobs.append((f"e5per|{n}",1,"zipf",p,32,kw,dict(WARM,persist_meta=True)))
for n,p,kw in CFG:
    jobs.append((f"e6no|{n}",1,"zipf",p,32,kw,dict(WARM,overlap=False)))
    jobs.append((f"e6ov|{n}",1,"zipf",p,32,kw,dict(WARM,overlap=True)))

print("jobs:",len(jobs),flush=True)
res={}
with ProcessPoolExecutor(max_workers=14) as ex:
    for i,(tag,r) in enumerate(ex.map(job,jobs),1):
        res[tag]=r
        if i%20==0: print(f"  {i}/{len(jobs)}",flush=True)
json.dump(res,open(Path(__file__).parents[1]/"data"/"results_v2.json","w"),indent=1,default=float)
print("done",len(res))
