# KV-Cache Offloading Architecture

**Distributed Endpoint Architecture for KV-Cache Offloading in LLM Inference**

---

## ⚠️ CONFIDENTIAL & PROPRIETARY

**© 2025 Subramaniyam (Sam) Pooni. All Rights Reserved.**

This documentation contains proprietary and confidential information. Unauthorized reproduction, distribution, or disclosure is strictly prohibited.

---

## Documentation Contents

| Section | Files | Description |
|---------|-------|-------------|
| Chapters | 14 | CH00-CH13 complete technical documentation |
| Appendix | 10 | A-J technical deep-dives |
| **Figures** | **51 source files (114 diagrams)** | Interactive visualizations |

## Key Results

| Metric | Value |
|--------|-------|
| Memory Expansion | 6× |
| User Capacity | 8× |
| HBM Hit Rate | 95% |
| Cost Reduction | 36% |
| TTFT Speedup | 15.6× |

## Figure Inventory (114 Total)

| Chapter | Diagrams | Description |
|---------|----------|-------------|
| CH00 | 14 | Executive Summary |
| CH01 | 5 | Introduction |
| CH02 | 8 | Background |
| CH03 | 11 | CXL Architecture |
| CH04 | 9 | Latency Analysis |
| CH05 | 5 | Bandwidth |
| CH06 | 3 | Preprocessing |
| CH07 | 27 | KV-Cache Management (Core) |
| CH08 | 5 | MoE Routing |
| CH09 | 7 | GPU Integration |
| CH10 | 6 | Market Landscape |
| CH11 | 6 | Performance |
| CH12 | 5 | Implementation |
| CH13 | 3 | Conclusion |

## File Structure

```
├── index.html                    # Main landing page
├── css/style.css                 # Shared stylesheet
├── chapters/                     # 14 chapter HTML files
├── appendix/                     # 10 appendix HTML files
└── figures/                      # 51 figure source files
    ├── index.html                # Figure gallery
    ├── CHAPTER-DIAGRAM-MAPPING.txt
    ├── ch00-executive-summary/   # 6 files (14 diagrams)
    ├── ch01-introduction/        # 3 files (5 diagrams)
    ├── ch02-background/          # 7 files (8 diagrams)
    ├── ch03-architecture/        # 5 files (11 diagrams)
    ├── ch04-latency/             # 4 files (9 diagrams)
    ├── ch05-bandwidth/           # 1 file (5 diagrams)
    ├── ch06-preprocessing/       # 1 file (3 diagrams)
    ├── ch07-kv-cache/            # 14 files (27 diagrams)
    ├── ch08-moe/                 # 1 file (5 diagrams)
    ├── ch09-gpu-integration/     # 2 files (7 diagrams)
    ├── ch10-market/              # 2 files (6 diagrams)
    ├── ch11-performance/         # 2 files (6 diagrams)
    ├── ch12-implementation/      # 2 files (5 diagrams)
    └── ch13-conclusion/          # 1 file (3 diagrams)
```

---

**© 2025 Subramaniyam (Sam) Pooni. All Rights Reserved.**

*Technical Documentation v3.0 — December 2025*
