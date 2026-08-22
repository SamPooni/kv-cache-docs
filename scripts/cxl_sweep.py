import numpy as np, json, importlib.util
spec=importlib.util.spec_from_file_location("m", Path(__file__).with_name("kv_tiering_sim_rev1_retained_for_audit.py"))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sh,bl,tu=m.build_workload(1)
cfgs=[("LRU","lru",{}),("LFU","lfu",{}),("EMA a=0.15","ema",{"alpha":0.15}),("EMA a=0.01","ema",{"alpha":0.01})]
res={}
print(f"{'CXL':>6} {'policy':>11} {'hit%':>7} {'cold%':>7} {'stall ms':>10}")
for cx in [32,64,128,256,512]:
    for name,pol,kw in cfgs:
        r=m.run(pol,32,sh,bl,tu,cxl_gib=cx,**kw)
        res[f"{cx}|{name}"]=r
        print(f"{cx:>5}G {name:>11} {r['hit_rate']:>7.2f} {r['cold_rate']:>7.2f} {r['mean_turn_stall_ms']:>10.1f}")
    print()
# reference byte table
GIB=1024**3
def kv(L,kvh,d,b): return 2*L*kvh*d*b
rows=[("Llama-2 70B (MHA, 64 KV heads), fp16",80,64,128,2),
      ("Llama-3 70B (GQA, 8 KV heads), fp16",80,8,128,2),
      ("Llama-3 70B (GQA), fp8",80,8,128,1),
      ("Llama-3 8B (GQA, 8 KV heads), fp16",32,8,128,2),
      ("Hypothetical MQA (1 KV head), fp16",80,1,128,2)]
print(f"{'config':>44} {'KiB/tok':>9} {'8K GiB':>8} {'128K GiB':>9}")
for n,L,h,d,b in rows:
    t=kv(L,h,d,b)
    print(f"{n:>44} {t/1024:>9.1f} {t*8192/GIB:>8.2f} {t*131072/GIB:>9.1f}")
json.dump({k:{kk:round(vv,3) if isinstance(vv,float) else vv for kk,vv in v.items()} for k,v in res.items()},open(Path(__file__).parents[1]/"data"/"cxl_sweep.json","w"),indent=1)
