# KV-Cache Offloading Architecture

**Distributed Endpoint Architecture for KV-Cache Offloading in LLM Inference**

---

## ⚠️ CONFIDENTIAL & PROPRIETARY

**© 2025 Subramaniyam (Sam) Pooni. All Rights Reserved.**

This documentation contains proprietary and confidential information belonging to Subramaniyam (Sam) Pooni. Unauthorized reproduction, distribution, or disclosure of this material is strictly prohibited without prior written consent.

---

📖 **[View Live Documentation](https://YOUR-USERNAME.github.io/kv-cache-docs/)**

## Key Results

| Metric | Improvement |
|--------|-------------|
| Memory Expansion | 6× |
| User Capacity | 8× |
| HBM Hit Rate | 95% |
| Cost Reduction | 36% |
| TTFT Speedup | 15.6× |
| Latency vs PCIe | 65× faster |

## Documentation Structure

```
├── index.html                    # Main landing page
├── css/style.css                 # Shared stylesheet
├── chapters/                     # 14 chapters
│   ├── ch00-executive-summary.html
│   ├── ch01-introduction.html
│   ├── ch02-background.html
│   ├── ch03-architecture.html
│   ├── ch04-latency.html
│   ├── ch05-bandwidth.html
│   ├── ch06-preprocessing.html
│   ├── ch07-kv-cache.html
│   ├── ch08-moe.html
│   ├── ch09-gpu-integration.html
│   ├── ch10-market.html
│   ├── ch11-performance.html
│   ├── ch12-implementation.html
│   └── ch13-conclusion.html
└── appendix/                     # 10 technical appendices
    ├── index.html
    ├── a-transformer-fundamentals.html
    ├── b-attention-mechanism.html
    ├── c-kv-cache-math.html
    ├── d-rope-encoding.html
    ├── e-head-specialization.html
    ├── f-cxl-technology.html
    ├── g-ema-algorithm.html
    ├── h-memory-hierarchy.html
    ├── i-bandwidth-calculations.html
    └── j-implementation-reference.html
```

## Core Innovations

1. **Per-Head Attention Tracking** — Track KV-cache importance at head granularity
2. **EMA-Based Scoring** — Exponential moving average captures sustained importance
3. **RoPE-Aware Prefetch** — Exploit position encoding locality
4. **Controller-Resident Intelligence** — ARM cores in endpoints manage cache autonomously

## Technology Stack

- CXL 3.0 for cache-coherent memory access
- Computational Storage Devices (CSDs) with ARM cores
- 256 GB DDR5 + 4 TB NVMe per endpoint
- 4 endpoints provide 1 TB expansion at 256 GB/s

---

**© 2025 Subramaniyam (Sam) Pooni. All Rights Reserved.**

*Technical Documentation v3.0 — December 2025*

*CONFIDENTIAL — Do not distribute without authorization*
