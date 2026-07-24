# Toy KV-Cache Demo for a Minimal Transformer

## Context

The user wants to build hands-on intuition for how LLM KV caching works by
implementing a minimal decoder-only (GPT-style) transformer from scratch and
comparing autoregressive generation performance with and without a KV cache.
The goal is understanding the systems/performance tradeoff: caching avoids
recomputing the whole prefix at every step. With standard full attention,
generation is still O(n²) overall with a KV cache, versus O(n³) attention
work when each step recomputes attention over the full prefix. This is not
about training a good language model —
random/untrained weights are sufficient since both code paths must produce
identical outputs by construction.

Decisions made with the user:
- **Language/framework:** Python + PyTorch, with attention and the KV-cache
  logic hand-written (not `nn.MultiheadAttention`) so the caching mechanism
  is fully visible.
- **Deliverable:** a single Jupyter notebook, interleaving code, brief
  explanations, and benchmark plots.
- **Location:** `~/kv-cache-demo` (a dedicated project folder, git-initialized
  separately from the user's home directory to avoid exposing unrelated
  credentials/config to a cloud session). No relevant deps found on the
  system (`torch`, `jupyter`, `matplotlib`, `notebook` all missing) — will
  set up a local virtual environment for isolation rather than installing
  into system Python.

## Implementation Plan

### 1. Environment setup
- Create a local venv, e.g. `python3 -m venv kvcache-venv`
- Install: `torch`, `notebook` (or `jupyter`), `matplotlib`, `ipykernel`
- Register the venv as a Jupyter kernel so the notebook uses it

### 2. Notebook: `kv_cache_demo.ipynb`

**a. Setup & tiny tokenizer**
- Character-level tokenizer over a small hardcoded text corpus (simplest,
  no external dataset needed), giving a small vocab (~60-100 tokens)

**b. Model definition**
- Config dataclass: `n_layers` (2-4), `d_model` (64-128), `n_heads` (2-4),
  `vocab_size`, `max_seq_len`
- Token + learned positional embeddings
- Hand-written causal self-attention block, written as **one function** used
  by both generation modes, with an optional `kv_cache` argument:
  - No cache: compute Q/K/V for the full sequence, apply causal mask, attend
  - With cache: compute Q/K/V for only the new token, append K/V to the
    passed-in cache tensors, attend new-token-Q over cached-K/V
- Small MLP block (linear → activation → linear), pre-norm residual
  connections (standard pre-LN transformer block)
- Final linear head to vocab logits
- Random weight init only — no training loop needed

**c. Two generation loops (same weights, same random seed)**
- `generate_no_cache(model, prompt_ids, n_new_tokens)`: at each step, run
  the full forward pass over the entire sequence generated so far
- `generate_with_cache(model, prompt_ids, n_new_tokens)`: maintain a
  per-layer `(K, V)` cache; each step forwards only the newest token,
  appends to cache, attends over cached + new

**d. Correctness check**
- Assert both generation paths produce identical logits/token ids for the
  same prompt and random seed — this is the check that the cache is
  mathematically equivalent, not just faster

**e. Benchmark**
- For increasing generation lengths (e.g. 32, 64, 128, 256, 512, 1024
  tokens), time both generation modes with `time.perf_counter()`
  (multiple repeats, take min/median to reduce noise)
- Record per-step time and cumulative time for each mode/length

**f. Plots (matplotlib)**
- Cumulative generation time vs sequence length, no-cache vs cache
  (expect cubic vs quadratic attention-work growth, although fixed overheads
  can obscure this at small sizes)
- Per-step time vs step index at a fixed target length (expect no-cache
  growing roughly quadratically with position, cache roughly linearly)

### 3. Verification
- Execute the notebook end-to-end (`jupyter nbconvert --execute` or running
  all cells) to confirm no errors and that plots render
- Confirm the correctness-check assertion passes (cache == no-cache outputs)
- Visually inspect the benchmark plots to confirm that caching avoids the
  repeated full-prefix computation at the chosen sequence lengths
