---
title: "Flood Detection from Satellite Imagery on the Edge"
date: 2026-06-14
draft: false
summary: "Lightweight segmentation models for near-real-time flood mapping from SAR and optical imagery, compressed to run on constrained on-board hardware."
tags: ["remote-sensing", "edge-ai", "segmentation"]
---

> This is an example project page. Replace the content with your own work — the
> structure below is a reasonable starting template.

## Overview

Near-real-time flood mapping needs results in minutes, not hours, which pushes
inference toward the sensor — on-board or at the ground station — rather than a
distant data center. This project explores compact segmentation models that keep
accuracy high while fitting tight compute, memory, and power budgets.

## Highlights

- Segmentation of flooded vs. non-flooded areas from SAR and optical inputs.
- Model compression (distillation + pruning) to shrink the network for the edge.
- Evaluation focused on the metrics that matter operationally: recall on flooded
  pixels and end-to-end latency.

## Stack

- **Modeling:** PyTorch
- **Data:** Sentinel-1 (SAR) and Sentinel-2 (optical)
- **Deployment target:** constrained on-board / edge hardware

## Links

- Repository: _add a link_
- Paper / report: _add a link_
