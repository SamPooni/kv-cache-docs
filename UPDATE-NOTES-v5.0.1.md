# v5.0.1 Update Notes

This maintenance revision tightens scientific language without changing the underlying GB10 measurements or Revision-2 simulation data.

## Changes

- Reclassifies the ~3.4× KV/fixed-path ratio as **model-derived from measured data**, not a direct per-byte measurement.
- Narrows the kernel conclusion to the tested scope: swapping between the two measured attention stacks did not eliminate the fitted KV slope.
- Replaces “compression dominates placement” with the supported conclusion that **reducing KV traffic is a first-order opportunity**. Compression, sparse/hierarchical selection, and placement act on different terms and require separate validation.
- Refreshes the package inventory to Appendix A–N and 58 current HTML figure sources.
- Removes the stale “93%+ hit rate” pitch claim.
- Reframes the front door around **KV-state management**, with CXL retained as a mechanism rather than the thesis.
- Keeps hierarchical relevance filtering / candidate selection out of the v5 result set; it is explicitly reserved for the next architecture revision.

No raw GB10 data, canonical regression coefficients, or Revision-2 simulation outputs were altered.
