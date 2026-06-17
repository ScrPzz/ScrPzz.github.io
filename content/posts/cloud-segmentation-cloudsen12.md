---
title: "Cloud segmentation on CloudSEN12: trees, superpixels, and maybe NAS"
date: 2026-06-17
draft: false
summary: "Setting up a benchmark for cloud/cloud-shadow segmentation on Sentinel-2 with CloudSEN12, from tree-based baselines to NAS."
tags: ["remote-sensing", "cloud-segmentation", "edge-ai", "nas"]
---

Setting up a new benchmark: **cloud and cloud-shadow segmentation on Sentinel-2**,
using [CloudSEN12](https://doi.org/10.1038/s41597-022-01878-2) (Aybar et al., 2022)
as reference — a global, hand-labeled set of ~49,400 chips (512×512, 11 S2 bands)
with four classes: *clear*, *thick cloud*, *thin cloud*, *cloud shadow*. The paper's
UNet baseline hits IoU ≈ 0.81 on cloud but only ≈ 0.62 on thin cloud — the hard class.

The question: how far can *lightweight, non-deep* methods go before a full
segmentation network is worth it? Plan, simplest first:

- **Tree-based, per-pixel** — gradient-boosted trees on the 11-band spectra plus a
  few indices. Fast and interpretable, but no spatial context — the floor.
- **Tree-based + superpixels** — SLIC segments, then classify each segment, to cut
  salt-and-pepper noise and recover thin-cloud edges.
- **NAS (maybe)** — search a compact architecture under an edge compute/memory budget,
  to see if it beats the trees enough to justify the cost.

The aim is an accuracy-vs-cost picture, with thin cloud and shadow as the deciding
classes. Numbers to follow — this is just the setup. *Placeholder, will update.*
