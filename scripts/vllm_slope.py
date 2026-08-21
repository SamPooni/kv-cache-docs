import time, json, argparse, os
from vllm import LLM, SamplingParams

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--params-b", type=float, default=7.6)
p.add_argument("--kv-dtype", default="auto", choices=["auto", "fp8"])
p.add_argument("--util", type=float, default=0.85)
p.add_argument("--maxlen", type=int, default=33000)
p.add_argument("--steps", type=int, default=32)
p.add_argument("--tag", default="fp16")
a = p.parse_args()

G = 2**30
llm = LLM(model=a.model, kv_cache_dtype=a.kv_dtype, max_model_len=a.maxlen,
          gpu_memory_utilization=a.util, enforce_eager=False,
          disable_log_stats=True)
cfg = llm.llm_engine.model_config.hf_config
hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
kvh = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
elt = 1 if a.kv_dtype == "fp8" else 2
bpt = 2 * cfg.num_hidden_layers * kvh * hd * elt
W = a.params_b * 1e9 * 2
print(f"\n[{a.tag}] KV/token {bpt/1024:.0f} KiB (elt={elt}B) | weights {W/G:.1f} GiB")

def run(B, L, n_out):
    ids = [[1000 + (i % 30000) for i in range(L)] for _ in range(B)]
    sp = SamplingParams(max_tokens=n_out, ignore_eos=True, temperature=0.0)
    prompts = [{"prompt_token_ids": x} for x in ids]
    t0 = time.perf_counter()
    llm.generate(prompts, sp, use_tqdm=False)
    return time.perf_counter() - t0

rows = []
print(f"{'batch':>6}{'ctx':>7}{'KVGiB':>8}{'ms/step':>9}{'tok/s':>8}")
for B, L in ((1, 4096), (1, 16384), (1, 32768), (4, 4096), (4, 16384),
             (4, 32768), (16, 4096), (16, 16384), (32, 4096)):
    kv = B * L * bpt
    try:
        run(B, L, 2)                                  # warm
        t1 = run(B, L, 2)
        tn = run(B, L, a.steps)
        ms = (tn - t1) / (a.steps - 2) * 1e3
        rows.append({"batch": B, "ctx": L, "kv_gib": kv / G, "ms": ms})
        print(f"{B:>6}{L:>7}{kv/G:>8.2f}{ms:>9.1f}{B/(ms/1e3):>8.1f}", flush=True)
    except Exception as e:
        print(f"{B:>6}{L:>7}  {str(e)[:50]}", flush=True)

n = len(rows)
if n >= 3:
    sx = sum(r["kv_gib"] for r in rows); sy = sum(r["ms"] for r in rows)
    sxy = sum(r["kv_gib"] * r["ms"] for r in rows); sxx = sum(r["kv_gib"]**2 for r in rows)
    slope = (n*sxy - sx*sy) / (n*sxx - sx*sx)
    icept = (sy - slope*sx) / n
    ybar = sy/n
    ssr = sum((r["ms"] - (icept + slope*r["kv_gib"]))**2 for r in rows)
    sst = sum((r["ms"] - ybar)**2 for r in rows)
    print(f"\n[{a.tag}] ms = {icept:.1f} + {slope:.2f} x KV_GiB   R2={1-ssr/sst:.3f}")
    print(f"[{a.tag}] KV bandwidth      {G/slope/1e6:.1f} GB/s")
    print(f"[{a.tag}] weight bandwidth  {W/icept/1e6:.1f} GB/s")
    json.dump({"tag": a.tag, "slope_ms_per_gib": slope, "intercept_ms": icept,
               "kv_bytes_per_token": bpt, "rows": rows},
              open(f"slope_{a.tag}.json", "w"), indent=1)
