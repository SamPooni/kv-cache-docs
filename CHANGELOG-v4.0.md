# Changelog — v4.0 (August 2026)

Consolidation pass answering the two external reviews (`kvevaluation.md`,
`kvcacheclaimsaudit.md`). Everything below was applied across all 74 HTML files,
the README and the figure indexes.

## 1. One canonical number set

`CANONICAL-NUMBERS.md` is now the single source of truth and is shipped with the
package. Appendix K reproduces every derivation. The aggressive set was retained
but made internally consistent and derivable:

| Metric | v3.0 | v4.0 | Where it now comes from |
|---|---|---|---|
| User capacity | 8× and 16× | **16×** | HBM hot set: 37 GB ÷ 16 users = 2.31 GB = 5.4% of a 43 GB context. Baseline is exactly 1 user (192 − 140 − 5 = 47 GB). |
| HBM hit rate | 97%, 95%, 93%, 85% | **97%** | Ladder 72 → 80 → 87 → 93 → 97 (+8/+7/+6/+4 percentage points) |
| CXL access latency | ~250 ns (over a table summing 260–310) | **200 ns** | Six components summing exactly to 200 |
| PCIe baseline | 5–10 µs / 16 µs / 13 µs | **13.00 µs** | Seven components summing exactly to 13.00 |
| Latency ratio | 40× / 52× / 65× | **65×** | 13,000 ÷ 200, with a mandatory scope statement |
| Effective latency | 231 ns / 504 µs / 1.0 ms / 0.50 ms | **177 ns** | 0.97×100 + 0.027×200 + 0.003×25,000; no recompute term |
| Cost reduction | 36% over an undisclosed BOM | **36%** | $160,000 → $103,000 CapEx table, both configs serving 16 users |
| EMA α | 0.1, 0.2 and 0.3 in different files | **0.2** | half-life ln0.5/ln0.8 = 3.1 decode steps |
| KV per user @128K | 41 GB | **43 GB** | 131,072 × 327,680 B = 42.95 GB = 40 GiB |
| Topology | 4 or 5 endpoints, 240/256/320 GB/s | **4 endpoints, 256 GB/s** | Capacity-driven: 687 GB into 1 TB |

## 2. Corrected errors

- **Appendix B GQA table** was 2× too low in every row. Now MHA 32 KB, GQA 4 KB,
  MQA 512 B per layer per token, reconciled to 320 KiB/token.
- **Appendix D / chapter 7 RoPE claim** `q_m·k_n ∝ cos((m−n)θ)` was wrong. Replaced
  with the sum over d/2 frequency pairs and the envelope bound `|⟨q_m,k_n⟩| ≤ B(|m−n|)`.
- **Appendix I** raised a 3.66 TB/s bandwidth objection against a 256 GB/s supply and
  dropped it with a non sequitur. It now refutes the strawman properly and derives
  the real steady-state demand: 11.4 GB/s, 4.5% of supply.
- **Appendix G** "+15% over LRU" withdrawn; EMA in isolation is +7 percentage points.
- **H100 → B200 migration** finished: 80 GiB HBM3 boxes, 1,280÷192 = "16×",
  8,000÷51 = "65×" and 40÷8,000 = "12 ms" all recomputed against real B200 numbers.
- **4,500 TFLOPS "dense"** is the 2:4-sparsity figure; dense BF16 is ~2,250 TFLOPS.
  Fixed everywhere, along with every derived arithmetic-intensity figure.
- **Metadata** 640 MB → 640 MiB (671 MB decimal) ≈ 1.5% of the 43 GB cache.
- **`ema-calculation.html`** computed at α=0.3 under an α=0.2 header;
  **`ema-eviction.html`** had four rows copy-pasted from the α=0.3 file. Both recomputed.
- **`figure-2-5-latency-waterfall.html`** shipped a source comment admitting the 65×
  gap. Comment removed and the waterfall made consistent.
- **`section-2-1-inference-fundamentals.html`** flagged 181 GB as "Overflow!" on a
  192 GB GPU. It fits; the figure now shows where it actually overflows.
- **`figure-3-2-memory-hierarchy.html`** titled "Three-Tier" over four tiers.
- **ch08 / ch09 section badges** read "Section 7" and "Section 8". Fixed.
- **`per-head-tracking.html`** used a different head taxonomy from the rest of the
  package. Unified on Recency / Anchor / Retrieval / Syntactic.
- **ch12 TCO panel** ("73% CapEx savings", "8 mo payback", a 4×B200 power line under
  an 8×B200 label, $2,500 endpoints) rebuilt from the canonical cost model.
- **Cover pages** disagreed (Version 1.0 vs 3.0, differing affiliations). Unified.

## 3. Prior art and market claims

- The fabricated competitor **"Niagara 2.0 / Astera Labs"** is removed. Astera's CXL
  line is Leo (memory controller), Scorpio (switch), Aries (retimer), Taurus.
- Unsubstantiated competitor rows (CXL-SpecKV, TraCT) removed rather than kept.
- "Nobody has", "first complete", "no existing solution", "The Innovation Gap" and
  "dumb DRAM" are gone from every file, including titles, nav labels and figure labels.
- The claim is now narrow and defensible: attention-score eviction is H2O, SnapKV,
  Scissorhands, PyramidKV and Quest; per-head budgets are Ada-KV and HeadKV; prefetch
  is InfiniGen; on-controller compute is Marvell Structera and Astera Leo. **What has
  not been publicly documented is per-head EMA scoring with RoPE-informed prefetch
  executed inside CXL controller firmware, below the serving framework.**
- Related-work blocks with arXiv links added to the landing page, both pitches, ch07,
  ch10 and the market figures. PNM-KV corrected to arXiv:2511.00321, November 2025.
- ch10 timeline extended into 2026; SGLang/RadixAttention and NVIDIA Dynamo promoted
  to first-class entries.

## 4. Evidence status

- **New Appendix K — Methodology, Assumptions and Evidence Status.** States plainly
  that nothing in the package is measured, reproduces every derivation, lists the
  load-bearing assumptions and what breaks if each is wrong, and specifies the
  validation programme that would convert the projections into results.
- Chapter 11 renamed from "Performance Benchmarks" to **"Performance Model and
  Projected Results"**, with threats-to-validity and what-would-validate-this sections.
- Evidence labels (Analytical model / Vendor specification / Design target /
  Illustrative) applied to quantitative claims throughout; every results figure carries
  a visible "analytical model, not measured" footer.
- Appendix F now separates CXL protocol semantics from implementation performance and
  discloses that published shipping-device latencies (Samsung CMM-D ~254 ns, CMM-B
  ~596 ns) are *higher* than the modeled 200 ns, with a sensitivity table showing what
  that does to the conclusions.
- "Zero CPU involvement" replaced everywhere with "no per-access CPU involvement in
  steady state", plus an explicit list of what host software still does.
- Direct GPU↔CXL.mem load/store is labelled a target-platform capability, not a
  generic CXL guarantee, in ch03, ch09 and the GPU figures.

## 5. Structure, prose and assets

- **Chapters rewritten**: 2,525 words across 14 chapters → **19,907 words**. Content
  that existed only inside figures (mailbox protocol, fault handling, market analysis,
  performance numbers) now lives in the chapters.
- ch08 updated from Mixtral-era 8 experts / top-2 to DeepSeek-V3-class 256 / top-8.
- **12 duplicate figure files removed**; all references repointed to a single copy.
- **Figures renumbered to match their chapters**: `figure-2-5-latency-waterfall` →
  `figure-4-3-…`; ch07's `figure-3-3/3-4/4-2/4-3` → `figure-7-2/7-3/7-9/7-10`;
  ch11's `figure-4-1/5-1` → `figure-11-1/11-2`.
- **React and Babel vendored locally** under `vendor/`. The 23 JSX figures previously
  loaded them from unpkg and rendered blank offline — exactly the setting a
  due-diligence reviewer opens them in. All 17 Babel blocks verified to compile.
- Missing `<meta charset>`, `<html lang>` and `<meta viewport>` added to the 6 files
  lacking them.
- Appendix navigation now exposes A–K on every appendix page (was A–D).
- `SUMMARY.txt`, `CHAPTER-DIAGRAM-MAPPING.txt` and all 14 `INDEX.txt` files are
  regenerated from the filesystem. The old "51 source files / 114 diagrams" counts
  were wrong; the real count is 44 figure files plus the gallery.
- 0 broken internal links; 0 unreferenced figure files; 0 HTML structural errors.

---

## v4.0.1 — follow-up pass (same day)

Three audit items were not closed in the first v4.0 build. Two are now fixed; one is a
deliberate decision, recorded here rather than left implied.

**Fixed**

- **Accessibility.** The audit found "zero `alt` text, `aria-*` or `role` attributes
  anywhere in 82 files". Now: all 48 `<iframe>` embeds carry a `title` derived from their
  figure label; every page has a `role="main"` landmark (74/74 files); 371 table headers
  carry `scope="col"`; data-bearing bar-chart fills carry `role="img"` with an
  `aria-label` stating the value; decorative brand icons are `aria-hidden`. This is a
  substantial improvement, not a claim of WCAG conformance — colour-coded diagram content
  inside the figures still lacks text equivalents.
- **"114 diagrams" survived in `figures/index.html`** while `SUMMARY.txt` said the count
  was unverified. Replaced with the verified 44 figure source files.
- **Fonts vendored.** All 38 files referenced Google Fonts over the network. The
  IBM Plex / Fraunces / Inter / JetBrains Mono woff2 files (804 KB) are now under
  `vendor/fonts/`. The package makes **no external requests of any kind** when opened;
  the only remaining `https://` URLs are citation hyperlinks the reader may choose to
  follow. Verified by loading the pack in a browser with networking disabled.

**Deliberately not done**

- **"54 of 82 files bypass `css/style.css`".** The 44 figure documents keep their own
  inline styles. They are standalone documents loaded in iframes, several are
  React-rendered, and each has a self-consistent visual system; retrofitting the shared
  stylesheet risks breaking working figures for cosmetic uniformity. The chapters,
  appendices and landing page — everything in the reading path — do use the shared
  stylesheet. Revisit only if the figures are rebuilt.
- **"Pre-transpile the 23 React files".** Solved differently: React and Babel are vendored
  locally, so the figures render offline without a build step. In-browser transpilation
  costs a few hundred milliseconds per figure and keeps the files editable as source.

---

## v4.0.2 — everything else the reviews asked for

The two items previously recorded as "deliberately not done", plus the remaining
unsourced constants, are now closed. Nothing from either review is outstanding except
the items that require hardware.

**One design system, applied**

- **All 44 figure documents now link `css/style.css`** and use its tokens for surfaces,
  text, borders, accents and type. The audit's "54 of 82 files bypass the stylesheet" is
  closed. Where a colour was doing fine-grained work inside a chart or gradient and no
  token matched, the literal value was kept deliberately rather than distorting a
  visualisation — those cases are noted in the source.
- **16 light-themed documents converted to the dark system.** The package used to flash
  white when a reader opened `ema-eviction`, `gqa-explainer`, `per-head-tracking`,
  `rope-prefetch`, `gpu-cxl-kvcache`, `market-landscape`, `competitive-landscape`,
  `cache-management-placement` and eight others. Pale pastel panels do not survive
  inversion, so each was rebuilt as a dark card with a low-alpha accent tint and a
  coloured left border.
- **`pitch.html` was visually broken and is fixed.** It linked the dark stylesheet while
  its own inline styles still carried a light palette from an earlier draft. The evidence
  statement — the most important paragraph in the pack — was rendering pale-grey-on-white
  at a measured **1.09:1** contrast ratio, i.e. invisible. It now measures **14.8:1**.
  `one-page-pitch.html` was converted to match.
- Font stacks were four different systems (Inter, system-ui, SF Pro Display, Segoe UI).
  All now resolve to the package's IBM Plex Sans / IBM Plex Mono / Fraunces tokens.

**Accessibility completed**

- **Every colour-coded figure now has a text equivalent**: 61 `sr-only` summaries, one per
  figure, each stating what the diagram shows and the specific values it encodes — not a
  generic placeholder. `role="img"` with a spoken `aria-label` on bars, tier blocks, score
  cells, timeline segments and comparison cells; decorative glyphs `aria-hidden`.
- **Zero WCAG AA contrast failures across 8,185 text nodes**, verified with an auditor that
  composites each text node against its real rendered background. The starting point was
  892 failures. Two palette bugs in `css/style.css` were the largest cause: `--text-muted`
  at #6e7681 failed AA on every surface in the package, and `.bar-fill` painted white text
  on light accent fills (as low as 1.68:1). Both fixed at the token level.

**Unsourced numbers closed**

- **51 GB/s vs 16.4 GB/s PCIe** reconciled: the first is the link-level uncontended figure,
  the second is what the model assumes an offload path sustains under contention. The
  0.32 contention factor is now named, and the consequence stated (at 51 GB/s the TTFT
  speedup would be 5.0×, not 15.6×).
- **The "2 TB/s switch backplane" is removed** — nothing derived or sourced it. The switch
  is now described by what the design requires of it: non-blocking across four ×16 Gen5
  downstream ports plus the host upstream port.
- **New Appendix K §K.7, "Uncalibrated parameters"** — a 13-row table covering β, the
  keep/demote/evict thresholds, prefetch windows, tier-placement weights, anchor-zone size,
  prefetch depth, eviction batch size and the rest: symbol, value used, what it controls,
  where it appears, and how it would be calibrated. Each figure that shows one of these
  now says it is uncalibrated and points at the table.
- The tier-placement score `P(p) = 0.25·R + 0.55·E + 0.20·N` and the prefetch priority
  `0.6 · rope + 0.4 · ema` are now explicitly distinguished in four places — they answer
  different questions and were readable as contradicting each other.
- **A seventh evidence label, `External literature`**, was added to the Appendix K taxonomy
  and applied to the 11 places where the package cites someone else's published result.
  It carries the same caveat as `Vendor specification`: evidence about another system.
- One real contradiction surfaced and was resolved: `figure-7-9-ema-eviction-policy.html`
  uses keep/evict thresholds of 0.6/0.25 against the design's 0.10/0.02. Both are now
  labelled, and the figure explains that its thresholds are fitted to a synthetic demo
  spread while a real EMA at α = 0.2 saturates near 0.04.

**Figure count, verified**

The old "114 diagrams" was reproducible from nothing. The package now states **61 figures
across 44 source files**, under a rule a reader can check: a figure is a block carrying its
own screen-reader summary. 43 files hold one; the ch12 diagram set holds 18.
`grep -c 'sr-only' figures/*/*.html` reproduces it.

**Verification on this build**

74 pages rendered in a fully offline browser: **0 JavaScript errors, 0 failed requests,
0 blank pages**. 0 broken internal links · 0 HTML structural errors · 17/17 Babel blocks
compile · 0 external network requests · 0 contrast failures · 0 iframes without a title ·
`role="main"` on 74/74 pages.
