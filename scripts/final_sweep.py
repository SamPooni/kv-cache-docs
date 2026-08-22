import numpy as np, json, importlib.util
spec = importlib.util.spec_from_file_location("m", Path(__file__).with_name("kv_tiering_sim_rev1_retained_for_audit.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
budgets=[8,16,32,64,128]
cfgs=[("LRU","lru",{}),("LFU","lfu",{}),("EMA a=0.15","ema",{"alpha":0.15}),("EMA a=0.01","ema",{"alpha":0.01})]
out={c[0]:{"hit":[],"err":[],"stall":[]} for c in cfgs}
wl=[m.build_workload(s) for s in (1,2)]
for hb in budgets:
    for name,pol,kw in cfgs:
        hs=[];st=[]
        for sh,bl,tu in wl:
            r=m.run(pol,hb,sh,bl,tu,**kw); hs.append(r["hit_rate"]); st.append(r["mean_turn_stall_ms"])
        out[name]["hit"].append(round(float(np.mean(hs)),2))
        out[name]["err"].append(round(float(np.std(hs)),2))
        out[name]["stall"].append(round(float(np.mean(st)),1))
        print(hb,name,out[name]["hit"][-1],flush=True)
json.dump({"budgets":budgets,"data":out},open(Path(__file__).parents[1]/"data"/"sweep_rev1_superseded.json","w"),indent=1)
