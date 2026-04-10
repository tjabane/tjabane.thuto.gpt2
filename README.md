# tjabane.thuto.gpt2

A GPT-2 style decoder-only transformer, built from scratch in Python.

## Architecture

A **decoder-only transformer** (the architecture behind GPT-2, GPT-3, and similar autoregressive language models) consists of a stack of identical blocks that each attend only to past tokens. There is no encoder and no cross-attention — the model learns to predict the next token given all previous ones.

![decoder only transformer architecture](image.png)

---

### Component Breakdown

#### 1. Input Embeddings
| Component | What it does |
|---|---|
| **Token Embedding** | Maps each token ID to a learned vector of size `d_model` |
| **Positional Embedding** | Adds a learned (GPT-2) or fixed (sinusoidal) vector encoding the token's position |

The two embeddings are summed to give the model both *what* the token is and *where* it sits in the sequence.

---

#### 2. Decoder Block × N

Each of the N identical blocks applies two sub-layers, each wrapped in a **pre-norm residual** connection:

```
x = x + Attn(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

##### Masked Multi-Head Self-Attention
- **Query / Key / Value** projections are computed from the *same* input (`Q = K = V = x`).
- A **causal mask** (upper-triangular −∞) ensures position `t` can only attend to positions `≤ t`, preserving the autoregressive property.
- Attention scores: `softmax(QKᵀ / √d_k) · V`
- Multiple heads learn different relational patterns in parallel; their outputs are concatenated and projected.

##### Feed-Forward Network (FFN)
A two-layer MLP applied position-wise:

```
FFN(x) = GELU(x W₁ + b₁) W₂ + b₂
```

The hidden dimension is typically **4× the model dimension** (e.g., 768 → 3072 in GPT-2 base).

##### Layer Norm & Residual
- **Pre-LayerNorm** (used in GPT-2): normalisation happens *before* each sub-layer.
- **Residual connections** let gradients flow directly to earlier layers, enabling deep stacks.

---

#### 3. Output Layer
| Component | What it does |
|---|---|
| **Final LayerNorm** | Normalises the last block's output before projection |
| **LM Head (Linear)** | Projects `d_model → vocab_size`; weights are **tied** to the token embedding matrix |
| **Softmax** | Converts logits to a probability distribution over the vocabulary |

The model is trained with **cross-entropy loss** on next-token prediction (teacher forcing). At inference, the next token is sampled from (or greedily selected from) the output distribution, then appended to the sequence and fed back in — this is **autoregressive generation**.

---

### Key Hyperparameters (GPT-2 Base)

| Parameter | Value |
|---|---|
| Layers (N) | 12 |
| Model dimension (d_model) | 768 |
| Attention heads | 12 |
| Head dimension (d_k) | 64 |
| FFN hidden size | 3072 |
| Vocabulary size | 50 257 |
| Max context length | 1 024 |
| Parameters | ~117 M |

---

### Why Decoder-Only?

- **No target sequence required** — the model trains on raw text with a simple next-token objective.
- **Scales well** — removing the encoder halves the architecture surface area, making very large models easier to train.
- **Flexible at inference** — prompting, few-shot, and chain-of-thought all emerge naturally from the autoregressive formulation.
