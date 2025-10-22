# Graph Transformer VAE for Link Prediction

> **Quick note:** Replace all `TODO:` items and file paths with your actual assets. All media paths are relative to the repo root.

---

## Table of Contents

* [Overview](#overview)
* [Architecture](#architecture)

  * [High-level Diagram](#high-level-diagram)
  * [Module Breakdown](#module-breakdown)
  * [Latent Inference & Decoding](#latent-inference--decoding)
* [Reproducing Results](#reproducing-results)
* [Visualizations](#visualizations)

  * [t-SNE (2D) Videos](#t-sne-2d-videos)
  * [Attention Head Maps](#attention-head-maps)
  * [3D t-SNE Latent Space](#3d-t-sne-latent-space)
* [Citations](#citations)
* [License](#license)

---

## Overview

A **Graph Transformer VAE** (GT-VAE) for link prediction. The encoder uses an adjacency-masked graph transformer with positional encodings; the decoder reconstructs edges from latent variables with a Bernoulli likelihood. This repo contains training code, evaluation scripts (AP/ROC), and visualization utilities.

> **Paper:** TODO: add arXiv/DOI link

---

## Architecture

### High-level Diagram

<p align="center">
  <img src="assets/figs/architecture/gt_vae_architecture.png" alt="GT-VAE Architecture" width="85%"/>
</p>

**Figure 1.** *GT-VAE*: (1) **Input Embedding** combines node features and Laplacian positional encodings; (2) **Adjacency‑Masked Graph Transformer Encoder** produces (\mu,,\log\sigma^2); (3) **Reparameterization** samples (\mathbf{z}); (4) **Decoder** scores edge existence via inner product (or MLP) and outputs (\hat{A}).

> **Add your diagram:** export from draw.io/PowerPoint as `assets/figs/architecture/gt_vae_architecture.png` (SVG/PNG preferred). Update the path above.

---

### Module Breakdown

**InputEmbedding**

* Projects node features `x ∈ R^{N×d_node}` and positional encodings `PE ∈ R^{N×d_pos}` to a shared hidden dimension and sums them:
  [ h_0 = W_x x + W_{pe} PE. ]

**Graph Transformer Encoder (Adjacency‑Masked)**

* Multi-head self-attention restricted by graph structure (mask by adjacency + self).
* Supports multi-layer feed-forward with residual + norm.
* Outputs (\mu, \log\sigma^2) for each node (or graph-level via pooling).

**Latent Variables**

* Reparameterization: (\mathbf{z} = \mu + \sigma \odot \epsilon,; \epsilon \sim \mathcal{N}(0, I)).

**Decoder**

* **Option A (Inner Product):** (\hat{A}_{ij} = \sigma(\langle z_i, z_j \rangle)).
* **Option B (MLP Edge Decoder):** concatenation `[z_i, z_j, |z_i−z_j|, z_i ⊙ z_j] → MLP → σ`.

**Loss**

* Balanced BCE over sampled edges/non-edges + KL:
  [ \mathcal{L} = \text{BCE}(\hat{A}, A) + \beta,\text{KL}(\mathcal{N}(\mu, \sigma^2) \parallel \mathcal{N}(0, I)). ]

---

### Latent Inference & Decoding

* **Validation/Test:** compute (\mu,\log\sigma^2) from the encoder; use (\mu) or samples for link scores.
* **Generation:** sample (\mathbf{z}) node-wise and decode edges; optionally enforce degree/graph priors.

---

## Reproducing Results

```bash
# 1) Setup
conda env create -f env.yml  # TODO: provide env.yml
conda activate gtn-vae

# 2) Train
python train.py \
  --dataset cora \
  --pos-enc laplacian --k-eigs 16 \
  --encoder gtn --layers 4 --heads 8 --hidden 128 \
  --beta 1.0 --batch-size 1 --epochs 400 \
  --logdir runs/cora

# 3) Evaluate
python eval.py --checkpoint checkpoints/cora/best.pt

# 4) Visualizations
python viz/tsne_video.py --checkpoint checkpoints/cora/best.pt
python viz/attention_maps.py --checkpoint checkpoints/cora/best.pt
python viz/tsne3d.py --checkpoint checkpoints/cora/best.pt
```

> **Tip:** Provide a minimal `scripts/download_data.sh` and `data/README.md` describing splits (RandomLinkSplit or custom) and seeds.

---

## Visualizations

### t-SNE (2D) Videos

> Place your rendered videos (or GIFs) in `assets/vis/tsne2d/`.

**Embed MP4 (works via raw HTML in GitHub Markdown):**

_<div align="center">
  <video src="assets/vis/tsne2d/cora_tsne_epoch_sweep.mp4" width="75%" controls muted loop></video>
</div>_

If autoplay is desired (may be blocked):

```html
<video src="assets/vis/tsne2d/cora_tsne_epoch_sweep.mp4" width="75%" autoplay muted loop playsinline></video>
```

**GIF fallback:**

<p align="center">
  <img src="assets/vis/tsne2d/cora_tsne_epoch_sweep.gif" width="75%" alt="2D t-SNE video (GIF)"/>
</p>

> **Naming suggestion:** `dataset_tsne_epoch_sweep.(mp4|gif)` or `dataset_tsne_classes.(mp4|gif)`.

---

### Attention Head Maps

> Save your attention heatmaps as PNG/SVG to `assets/vis/attn/` using the convention `layer{L}_head{H}.png`.

**Single-layer grid (example for L=0):**

<p align="center">
  <img src="assets/vis/attn/layer0_grid.png" width="90%" alt="Attention heads layer 0"/>
</p>

**Expandable layers:**

<details>
  <summary><b>Layer 0</b></summary>
  <img src="assets/vis/attn/layer0_grid.png" width="95%"/>
</details>
<details>
  <summary><b>Layer 1</b></summary>
  <img src="assets/vis/attn/layer1_grid.png" width="95%"/>
</details>
<details>
  <summary><b>Layer 2</b></summary>
  <img src="assets/vis/attn/layer2_grid.png" width="95%"/>
</details>

> **Tip:** Include a short legend explaining masking (adjacency + self) and color scale (low→high). Consider adding per-head sparsity metrics in captions.

---

### 3D t-SNE Latent Space

> Export a static render and (optionally) an interactive HTML.

**Static preview:**

<p align="center">
  <img src="tsne_viz_3d.gif" width="75%" alt="3D t-SNE video (GIF)"/>
</p>

**Interactive (Plotly HTML):**

> GitHub does not render `.html` inline; link to the file and to a short screen capture.

* Download the interactive file: [`assets/vis/tsne3d/cora_tsne3d.html`](assets/vis/tsne3d/cora_tsne3d.html)
* Short demo video:

<div align="center">
  <video src="assets/vis/tsne3d/cora_tsne3d_demo.mp4" width="70%" controls muted loop></video>
</div>

**Optional iframe for GitHub Pages:** If you publish docs via GitHub Pages, embed the HTML interactively there and link to it:

```html
<!-- docs/tsne3d.html hosts the Plotly figure; add your site URL below -->
<iframe
  src="https://USER.github.io/REPO/docs/tsne3d.html"
  width="100%" height="520" frameborder="0"></iframe>
```

---
