# RVQ-BeT: Residual Vector-Quantized Action Tokenization for Behavior Transformer

This repository contains a course-project implementation that modifies the original **Behavior Transformer (BeT)** action discretization pipeline on the **BlockPush** environment. The main goal is to replace BeT's original K-means action tokenizer with a learned **Residual VQ-VAE / RVQ-based action tokenizer**, inspired by VQ-BeT, while keeping the GPT-like Transformer policy backbone unchanged.

The project is designed as a controlled empirical study rather than a full reimplementation of VQ-BeT. The central question is:

> Can a learned VQ/RVQ action tokenizer improve or match the original K-means-based BeT policy on a multimodal continuous-control task?

---

## 1. Project Motivation

Original BeT discretizes continuous actions using K-means. Each action is assigned to the nearest cluster center, and the Transformer predicts the corresponding discrete action bin. A continuous offset is then predicted to refine the selected action center.

This project studies whether replacing the fixed K-means tokenizer with a learned VQ/RVQ tokenizer can improve action representation and behavior generation.

The comparison is intentionally controlled:

- The **Transformer backbone is kept unchanged**.
- The main modification is the **action tokenizer** and the corresponding output heads.
- The number of discrete codes is fixed to **K = 16** for all K-means and RVQ variants.
- RVQ is evaluated with both **N_q = 1** and **N_q = 2**.

---

## 2. Implemented Variants

The following variants are implemented and evaluated:

| Variant | Tokenizer | N_q | K | Offset Mode | Description |
|---|---:|---:|---:|---|---|
| `kmeans_k16` | K-means | - | 16 | code-dependent | Original BeT-style baseline |
| `rvq_nq1_k16_code_independent_offset` | VQ/RVQ | 1 | 16 | code-independent | Single offset from hidden state |
| `rvq_nq1_k16_code_dependent_offset` | VQ/RVQ | 1 | 16 | code-dependent | One offset branch per code |
| `rvq_nq2_k16_code_independent_offset` | RVQ | 2 | 16/layer | code-independent | One offset from hidden state |
| `rvq_nq2_k16_primary_dependent_offset` | RVQ | 2 | 16/layer | primary-code-dependent | Offset branch selected by primary code |
| `rvq_nq2_k16_code_embedding_conditioned_offset` | RVQ | 2 | 16/layer | code-embedding-conditioned | Offset predicted from hidden state and selected code embeddings |

---

## 3. Method Overview

### 3.1 Original BeT Action Prediction

Original BeT uses K-means to discretize continuous actions:

\[
c = \arg\min_k \|a - A_k\|_2^2
\]

where \(A_k\) is the K-means action center. The policy predicts the discrete action bin and a residual offset:

\[
\hat a = A_{\hat c} + \Delta \hat a_{\hat c}
\]

This project keeps this baseline with:

\[
K = 16
\]

---

### 3.2 RVQ Action Tokenizer

The RVQ tokenizer encodes each action into one or more discrete latent codes.

For \(N_q=1\):

\[
a \rightarrow z^{(1)}
\]

For \(N_q=2\):

\[
a \rightarrow (z^{(1)}, z^{(2)})
\]

The quantized latent representation is reconstructed as:

\[
z_q = \sum_{i=1}^{N_q} e^{(i)}_{z^{(i)}}
\]

and decoded into an action center:

\[
a_{center} = \psi(z_q)
\]

The final action is:

\[
\hat a = a_{center} + \Delta \hat a
\]

The RVQ codebooks are initialized with K-means to reduce code collapse. Code usage, active code count, and perplexity are logged for diagnosis.

---

## 4. Policy Architecture

The GPT-like Transformer backbone is not modified. In particular, the following components are kept unchanged:

- input embedding;
- positional embedding;
- attention blocks;
- Transformer MLP blocks;
- hidden size;
- number of layers;
- number of heads;
- block size.

Only the output heads and loss computation are modified to support RVQ codes.

For \(N_q=1\):

\[
h_t = \mathrm{Transformer}(o_{t-h:t})
\]

\[
\mathrm{logits}^{(1)}_t = W_1 h_t
\]

For \(N_q=2\):

\[
\mathrm{logits}^{(1)}_t = W_1 h_t,\quad
\mathrm{logits}^{(2)}_t = W_2 h_t
\]

The code prediction loss is:

\[
L_{code} = L_1 + \beta L_2
\]

where \(L_1\) is the primary-code classification loss and \(L_2\) is the secondary-code classification loss. In the current experiments, equal weighting is used unless otherwise specified.

---

## 5. Offset Modes

### 5.1 Code-independent Offset

A single offset is predicted from the Transformer hidden state:

\[
\Delta \hat a = f(h_t)
\]

This mode does not explicitly condition on the predicted code.

---

### 5.2 Code-dependent Offset

For \(N_q=1\), the model predicts one offset candidate for each code:

\[
[\Delta \hat a_1, \ldots, \Delta \hat a_K] = f(h_t)
\]

The offset corresponding to the target or sampled code is selected:

\[
\Delta \hat a = \Delta \hat a_{\hat z}
\]

This is closest to the original BeT offset design.

---

### 5.3 Primary-code-dependent Offset

For \(N_q=2\), the model predicts offset candidates conditioned only on the first RVQ layer:

\[
[\Delta \hat a_1, \ldots, \Delta \hat a_K] = f(h_t)
\]

The branch is selected using the primary code \(z^{(1)}\), not the full code pair:

\[
\Delta \hat a = \Delta \hat a_{z^{(1)}}
\]

This avoids a full \(K^2 \cdot d_a\) output head.

---

### 5.4 Code-embedding-conditioned Offset

For \(N_q=2\), the selected code embeddings are concatenated with the Transformer hidden state:

\[
\Delta \hat a = f(h_t, e^{(1)}_{z^{(1)}}, e^{(2)}_{z^{(2)}})
\]

During training, ground-truth RVQ codes are used as conditions. During rollout, sampled or argmax-predicted codes are used.

---

## 6. Loss Function

For RVQ \(N_q=2\), the total loss is:

\[
L = L_1 + \beta L_2 + \lambda_{offset} L_{offset}
\]

where:

\[
L_i = \mathrm{FocalLoss}_{\gamma}(\mathrm{logits}^{(i)}, z^{(i)})
\]

In the current configurations, \(\gamma = 2.0\). When \(\gamma = 0\), this reduces to the standard cross-entropy loss.

and:

\[
L_{offset} = \|\Delta \hat a - \Delta a\|^2
\]

The offset target is defined as:

\[
\Delta a = a - \psi(e^{(1)}_{z^{(1)}} + e^{(2)}_{z^{(2)}})
\]

For \(N_q=1\), the same form applies with only one code layer.

---

## 7. Evaluation Protocol

The main evaluation environment is **BlockPush**. Evaluation uses 50 rollout episodes unless otherwise noted.

Example evaluation command:

```bash
WANDB_MODE=offline python -u -B run_on_env.py \
  --config-name=eval_blockpush \
  load_dir=/path/to/experiment_dir \
  window_size=5 \
  enable_render=False \
  num_eval_eps=50
```

Evaluation-time code selection can be controlled with:

```bash
latent_sampling_strategy=argmax
```

or:

```bash
latent_sampling_strategy=sampling
```

For RVQ policies, leaving `latent_sampling_strategy` unset uses the strategy saved in the state prior configuration, which is currently `rvq_sampling_strategy: argmax` for the RVQ configs.

Example diagnostic command:

```bash
conda run -n behavior-transformer python -u -B tests/diagnose_rvq_snapshot.py \
  --snapshot /path/to/snapshot.pt \
  --window-size 5 \
  --split val \
  --max-windows 4096 \
  --device cuda
```

---

## 8. Main Results

The following results are from 50-episode BlockPush evaluation runs.

| Model | Average Reward | Std | Interpretation |
|---|---:|---:|---|
| K-means K=16 baseline | 0.8564 | 0.2487 | Strong original BeT baseline |
| RVQ N_q=1 + code-dependent offset | 0.8776 | 0.2769 | Best observed result; slightly above K-means |
| RVQ N_q=2 + primary-dependent offset | 0.6768 | 0.3852 | Clearly worse than baseline |
| RVQ N_q=2 + code-embedding-conditioned offset | 0.7266 | 0.3511 | Better than primary-dependent, but still below baseline |

The main empirical finding is that **RVQ N_q=1 with code-dependent offset performs comparably to, and slightly better than, the K-means BeT baseline**, while the tested \(N_q=2\) variants underperform.

---

## 9. RVQ Diagnostic Results

Diagnostics show that the RVQ tokenizer itself does **not** collapse.

For the \(N_q=2\) tokenizer:

| Layer | Active Codes | Perplexity | Max Usage Ratio |
|---|---:|---:|---:|
| q0 / primary | 16/16 | 13.98 | 0.1110 |
| q1 / secondary | 16/16 | 11.19 | 0.1875 |

Tokenizer reconstruction is also reasonable:

| Metric | Value |
|---|---:|
| center L1 | 0.0007257 |
| center MSE | 9.81e-07 |
| offset target abs mean | 0.0007257 |

Therefore, the weaker \(N_q=2\) rollout performance is not caused by code collapse.

---

## 10. Code Prediction Diagnostics

For \(N_q=2\), the policy must predict two codes. Validation diagnostics show that the second-layer residual code is significantly harder to predict.

| Model | Acc(z1) | Acc@3(z1) | Acc(z2) | Acc@3(z2) | Joint Acc(z1,z2) |
|---|---:|---:|---:|---:|---:|
| N_q=2 primary-dependent offset | 0.8075 | 0.9615 | 0.5719 | 0.8694 | 0.5415 |
| N_q=2 code-embedding-conditioned offset | 0.8013 | 0.9592 | 0.5682 | 0.8642 | 0.5386 |

The primary code is predicted reasonably well, but the secondary code remains much less accurate. Since the RVQ decoder requires the full pair \((z^{(1)}, z^{(2)})\), the joint code accuracy around 54% is likely a major bottleneck.

This explains why \(N_q=2\) underperforms despite having a non-collapsed tokenizer.

---

## 11. Interpretation

The experiments suggest three main conclusions:

1. **Learned VQ tokenization is feasible inside the BeT framework.**
   The \(N_q=1\) RVQ/VQ model with code-dependent offset reaches performance comparable to the original K-means baseline.

2. **Offset parameterization is crucial.**
   Code-independent offsets are too weak in this setting, while code-dependent or code-conditioned offsets provide more useful residual correction.

3. **More RVQ layers are not automatically better.**
   Although \(N_q=2\) gives a richer hierarchical action representation, it also increases code prediction difficulty. For low-dimensional BlockPush actions, this additional complexity currently hurts rollout performance.

---

## 12. Suggested Future Work

Several follow-up experiments are natural:

1. **Sampling-vs-argmax rollout comparison**
   Argmax rollout is implemented through `latent_sampling_strategy=argmax`. A natural follow-up is to systematically compare argmax and categorical sampling across variants and seeds.

2. **Secondary-code loss reweighting**
   Try:

   \[
   L_{code} = L_1 + \beta L_2
   \]

   with:

   \[
   \beta \in \{0.1, 0.25, 0.5, 1.0\}
   \]

   to reduce overemphasis on the hard-to-predict secondary code.

3. **Predicted-code action error**
   Measure action reconstruction error using predicted codes rather than ground-truth tokenizer codes:

   \[
   \|a - \psi(e_{\hat z^{(1)}} + e_{\hat z^{(2)}})\|
   \]

4. **Code-pair validity mask**
   Prevent the model from sampling unseen or rare \((z^{(1)}, z^{(2)})\) pairs.

5. **Trajectory visualization**
   Plot agent and block trajectories for qualitative comparison of K-means and RVQ policies.

6. **Multi-seed evaluation**
   Current reward differences, especially between K-means and \(N_q=1\), should be validated across multiple random seeds.

---

## 13. Project Status

Current status:

- K-means K=16 baseline implemented.
- RVQ tokenizer with K=16 and N_q=1/2 implemented.
- K-means initialization for RVQ codebooks implemented.
- Code usage and perplexity diagnostics implemented.
- N_q=1 code-dependent offset implemented.
- N_q=2 primary-code-dependent offset implemented.
- N_q=2 code-embedding-conditioned offset implemented.
- Snapshot-based RVQ diagnostics implemented.

Current best-performing variant:

```text
RVQ N_q=1, K=16, code-dependent offset
```

Primary limitation:

```text
N_q=2 introduces a difficult secondary-code prediction problem, reducing rollout stability.
```

---

## 14. Notes

This project should be understood as a **VQ-BeT-inspired modification of BeT**, not a complete reproduction of VQ-BeT. The main contribution is a controlled comparison between K-means action tokenization and learned RVQ action tokenization under an otherwise mostly unchanged BeT policy architecture.
