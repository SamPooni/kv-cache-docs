import time, argparse, torch
from transformers import AutoConfig, AutoModelForCausalLM

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--params-b", type=float, default=7.6)
p.add_argument("--spec-bw", type=float, default=273.0)
p.add_argument("--budget-gib", type=float, default=90.0)
p.add_argument("--steps", type=int, default=20)
p.add_argument("--dry-run", action="store_true")
a = p.parse_args()

GiB = 2**30
sync = torch.cuda.synchronize

def measure_bw(gib=4.0):
    n = int(gib * GiB) // 2
    x = torch.ones(n, dtype=torch.float16, device="cuda")
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    sync()
    t = []
    for _ in range(20):
        s = time.perf_counter(); y.copy_(x); sync()
        t.append(time.perf_counter() - s)
    t.sort()
    del x, y
    torch.cuda.empty_cache()
    return 2 * n * 2 / t[10] / 1e9

if a.dry_run:
    B = a.spec_bw * 0.8 * 1e9
    print(f"streaming bandwidth : {B/1e9:.1f} GB/s  (assumed, dry run)")
else:
    g = measure_bw()
    B = g * 1e9
    print(f"streaming bandwidth : {g:.1f} GB/s  ({100*g/a.spec_bw:.0f}% of {a.spec_bw:.0f} spec)")

c = AutoConfig.from_pretrained(a.model, trust_remote_code=True)
hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
kvh = getattr(c, "num_key_value_heads", None) or c.num_attention_heads
bpt = 2 * c.num_hidden_layers * kvh * hd * 2
W = a.params_b * 1e9 * 2
print(f"model  : {a.model}")
print(f"  {c.num_hidden_layers} layers, {kvh} kv heads, head_dim {hd}")
print(f"  KV/token {bpt/1024:.0f} KiB | weights {W/GiB:.1f} GiB | "
      f"weight-only floor {B/W:.1f} tok/s\n")

grid = []
for b in (1, 4, 16, 32):
    for x in (4096, 16384, 32768):
        if (b * x * bpt + W) / GiB <= a.budget_gib:
            grid.append((b, x))

hdr = f"{'batch':>6}{'ctx':>7}{'KV GiB':>9}{'pred ms':>9}{'meas ms':>9}{'ratio':>7}{'tok/s':>8}{'KV%':>6}"
print(hdr)

if a.dry_run:
    for b, x in grid:
        kv = b * x * bpt
        t = (W + kv) / B
        print(f"{b:>6}{x:>7}{kv/GiB:>9.2f}{t*1e3:>9.1f}{'-':>9}{'-':>7}"
              f"{b/t:>8.1f}{100*kv/(kv+W):>5.0f}%")
    raise SystemExit

m = AutoModelForCausalLM.from_pretrained(
    a.model, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True).eval()

for b, x in grid:
    kv = b * x * bpt
    pred_ms = (W + kv) / B * 1e3
    try:
        ids = torch.randint(0, c.vocab_size, (b, x), device="cuda")
        with torch.inference_mode():
            past = m(ids, use_cache=True).past_key_values
            nx = torch.randint(0, c.vocab_size, (b, 1), device="cuda")
            for _ in range(3):
                past = m(nx, past_key_values=past, use_cache=True).past_key_values
            sync()
            t = []
            for _ in range(a.steps):
                s = time.perf_counter()
                past = m(nx, past_key_values=past, use_cache=True).past_key_values
                sync()
                t.append(time.perf_counter() - s)
        t.sort()
        ms = t[len(t)//2] * 1e3
        print(f"{b:>6}{x:>7}{kv/GiB:>9.2f}{pred_ms:>9.1f}{ms:>9.1f}"
              f"{ms/pred_ms:>7.2f}{b/(ms/1e3):>8.1f}{100*kv/(kv+W):>5.0f}%")
        del ids, past
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"{b:>6}{x:>7}   {str(e)[:44]}")
        torch.cuda.empty_cache()
