# Measured: the KV bandwidth wall on GB10

**Status: measured on hardware.** All numbers in this document come from a DGX Spark
(GB10 Grace Blackwell, 128 GiB coherent unified LPDDR5X, 273 GB/s spec), 19 August 2026.
Nothing here is simulated.

Model under test: Qwen2.5-7B-Instruct, fp16 — 28 layers, 4 KV heads, head_dim 128.
KV = **56 KiB/token**. Weights = 15.2 GB = **14.16 GiB**.

---

## 1. Achieved memory bandwidth

Streaming `copy` kernel (read + write), 4 GiB buffers, median of 20:

```
measured   236.5 GB/s          = 87% of the 273 GB/s spec figure
```

This replaces the *assumed* effective bandwidth used in earlier drafts, which applied an
undefended ~50% haircut to a theoretical figure. The haircut was wrong: real achieved
bandwidth is 87% of spec, not 50%.

---

## 2. The decode model is two-term, not one-term

The single-bandwidth roofline `t = (W + KV)/B` does not hold. Sweeping resident KV volume
across batch × context and fitting `t = a + b·KV` gives:

| Framework | Fit | n | R² | Weight bandwidth | KV bandwidth |
|---|---|---|---|---|---|
| HuggingFace + SDPA, unpaged | `t = 100.3 + 16.34·KV_GiB` ms | 7 | **0.991** | 151.5 GB/s | **65.7 GB/s** |
| vLLM + FlashAttention-2, paged | `t = 73.9 + 17.58·KV_GiB` ms | 9 | 0.945 | 205.7 GB/s | **61.1 GB/s** |

As a fraction of the 236.5 GB/s the hardware actually achieves:

```
weight read   HF   64%          KV read   HF   28%
              vLLM 87%                    vLLM 26%
```

### The result

> **KV is read at roughly a quarter of achievable memory bandwidth in the measured GB10 runs, and substituting between the two tested attention stacks did not eliminate that slope.**

Two independent frameworks — different attention implementations, paged versus unpaged KV
layout — produce KV slopes **within 8% of each other** (16.34 vs 17.58 ms/GiB). Over the
same change, the *weight* read improves by 34% (151.5 → 205.7 GB/s), reaching 87% of
achievable bandwidth.

vLLM extracts near-peak bandwidth on weights and a quarter of it on KV.

**Under the two-term GB10 fit, the inferred effective KV path has ~3.4× higher per-byte cost than the fixed path** (205.7 / 61.1). This ratio is model-derived, not a direct byte-level hardware measurement.

---

## 3. What this settles

**KV traffic reduction is the first-order opportunity exposed by this measurement.** Compression reduces bytes per KV element; sparse or hierarchical selection can reduce the number of KV elements consumed. Placement remains necessary for capacity, reuse, and movement. The relative end-to-end benefit of these mechanisms has not yet been established.

**Kernel substitution alone did not eliminate the measured KV slope.** FlashAttention-2 with paged KV and SDPA with contiguous KV produced similar slopes on this GB10/Qwen2.5-7B test. This does not rule out gains from other kernels, GPU architectures, models, page sizes, or decode regimes.

**Capacity was never the binding constraint here.** Every configuration measured fit
comfortably in 128 GiB. Throughput still collapsed: 150 tok/s at batch 16 × 4K falls to
46 tok/s at batch 16 × 16K, with memory to spare and nothing evicted.

---

## 4. Raw data

### HuggingFace + SDPA (fp16, chunked prefill, unpaged)

| batch | ctx | KV GiB | measured ms/step | tok/s | KV % of bytes |
|---|---|---|---|---|---|
| 1 | 4096 | 0.22 | 104.6 | 9.6 | 2% |
| 1 | 16384 | 0.88 | 113.9 | 8.8 | 6% |
| 1 | 32768 | 1.75 | 128.5 | 7.8 | 11% |
| 4 | 4096 | 0.88 | 111.8 | 35.8 | 6% |
| 4 | 16384 | 3.50 | 155.4 | 25.7 | 20% |
| 4 | 32768 | 7.00 | 212.4 | 18.8 | 33% |
| 16 | 4096 | 3.50 | 165.1 | 96.9 | 20% |

### vLLM 0.20.1 + FlashAttention-2 (fp16, paged, prefix caching disabled)

| batch | ctx | KV GiB | measured ms/step | tok/s |
|---|---|---|---|---|
| 1 | 4096 | 0.22 | 95.1 | 10.5 |
| 1 | 16384 | 0.88 | 101.5 | 9.8 |
| 1 | 32000 | 1.71 | 107.8 | 9.3 |
| 4 | 4096 | 0.88 | 99.9 | 40.0 |
| 4 | 16384 | 3.50 | 117.4 | 34.1 |
| 4 | 32000 | 6.84 | 162.2 | 24.7 |
| 16 | 4096 | 3.50 | 127.5 | 125.5 |
| 16 | 16384 | 14.00 | 345.9 | 46.2 |
| 32 | 4096 | 7.00 | 185.0 | 173.0 |

---

## 5. Method

Decode step time is isolated by differencing: `(t(N) − t(2)) / (N − 2)` with N = 32, which
cancels prefill and warm-up. Each configuration is preceded by an untimed warm run. Medians
of 20–30 steps. Batch sequences carry distinct token ids so no KV is shared between them.

**Two measurement bugs were found and fixed during this work; both are recorded because both
produced plausible-looking wrong answers.**

1. **fp32 load.** `torch_dtype=` is deprecated in current transformers and was silently
   ignored, loading weights in fp32 (28.4 GiB) and triggering the OOM killer. Unified memory
   surfaces a CUDA OOM as a Linux OOM kill, which obscures the cause. Fixed with `dtype=`;
   the script now prints resident weight bytes and dtype to make a repeat visible.

2. **Prefix-sharing artifact.** The first vLLM run generated identical token ids for every
   sequence in the batch. vLLM's prefix caching stored one shared copy of the KV, so true
   resident KV was `L × bpt`, not `B × L × bpt` — inflating the x-axis by a factor of B and
   producing an apparent KV bandwidth of **342.9 GB/s**. That figure exceeds the 273 GB/s
   spec, which is what exposed it: a measured read bandwidth above the memory bus is
   impossible. Fixed by giving each sequence distinct tokens and setting
   `enable_prefix_caching=False`.

The second bug is the useful one to remember. It did not produce an obviously broken number —
it produced a *better* number, one that would have supported a more exciting conclusion
("vLLM is 5× more efficient on KV"). It was caught only by checking the result against a
physical bound.

---

## 6. Limitations

- **One model, one size, one machine.** Qwen2.5-7B on a single GB10. No claim about H100,
  MI300, or larger models.
- **Fixed-length uniform batches.** No continuous batching with mixed arrival, no ragged
  sequence lengths.
- **The vLLM fit is noisier** (R² 0.945 vs 0.991), and the 16×16384 point sits 26 ms above
  the fit. Needs a repeat before that point is relied on.
- **No thermal log was captured** for these runs. EC firmware is `0x03000508`, past the
  versions reported to throttle, and idle temps were 45–49 °C — but sustained-load clocks
  were not recorded. A repeat should log `clocks.sm` alongside.
- **FlashAttention-2, not 3.** vLLM selected FA2 on this platform. FA3 or a flash-decoding
  split-K path may behave differently at batch 1, where a single query attends to 32K keys
  with little parallelism to exploit.
- **fp8 KV not yet measured.** That run is next and is the direct test of the compression
  claim.

---

## 7. Next measurement

`kv_cache_dtype=fp8` halves KV bytes per token (56 → 28 KiB). Two outcomes:

- Slope holds near 17.6 ms/GiB while GiB halve → **~2× on the KV term**, confirming that
  bytes are what matter and compression is the lever.
- Slope roughly doubles to ~35 ms/GiB → dequantisation cost cancels the byte saving, and
  fp8 buys capacity but not bandwidth on this hardware.

Either outcome is a result. The second would be the more interesting one.
