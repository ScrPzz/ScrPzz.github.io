---
title: "Hello, and what this blog is about"
date: 2026-06-14
draft: false
summary: "A short intro to the blog and the topics I'll write about — model compression, NAS, and edge AI for satellite remote sensing."
tags: ["meta", "edge-ai", "knowledge-distillation"]
categories: ["notes"]
math: true
---

Welcome! I'm Alessandro, a data scientist at the Italian Space Agency (ASI).
This is where I'll write about making deep-learning models small, fast, and
good enough to run far from a data center — including in orbit.

## What you'll find here

- **Knowledge distillation** — training compact *student* models to inherit the
  behaviour of larger *teacher* models.
- **Neural architecture search (NAS)** — letting the search process, rather than
  intuition, find architectures under tight compute and memory budgets.
- **Edge AI for remote sensing** — running inference on constrained on-board
  hardware, where bandwidth, power, and latency all matter.

## A taste: the distillation objective

Knowledge distillation trains a small student to match a larger teacher. A common
objective blends a supervised cross-entropy term with a soft-target term:

$$
\mathcal{L} = (1-\alpha)\,\mathcal{L}_{\text{CE}}\big(y, \sigma(z_s)\big)
            + \alpha\,T^{2}\,\mathrm{KL}\!\left(\sigma(z_t/T)\,\|\,\sigma(z_s/T)\right)
$$

where \(z_s\) and \(z_t\) are the student and teacher logits, \(T\) is the
temperature, and \(\alpha\) balances the two terms. Inline math such as \(T^{2}\)
renders too.

## Code blocks work as well

```python
import torch.nn.functional as F

def distill_loss(student_logits, teacher_logits, targets, T=4.0, alpha=0.5):
    ce = F.cross_entropy(student_logits, targets)
    kl = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)
    return (1 - alpha) * ce + alpha * kl
```

More soon. Feel free to replace this post with your own first one — it doubles as
a quick check that math, code highlighting, and the table of contents all work.
