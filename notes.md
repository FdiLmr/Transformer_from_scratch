# Transformer from Scratch
---

## Recurrent Neural Networks (RNN)
RNNs process sequential data by maintaining a state that is passed through time steps:
- Input: \( X_1, X_2, ..., X_N \)
- Outputs: \( Y_1, Y_2, ..., Y_N \)
- Challenges:
  1. Slow computation for long sequences.
  2. Vanishing or exploding gradients.
  3. Difficulty accessing long-term dependencies.

---

## Transformers
Transformers address RNN limitations using:
- Parallelized computations.
- Attention mechanisms for long-range dependency tracking.

### Notations:
- Input matrix: \( (sequence, d_{model}) \)

---

## Encoder
The Encoder module processes the input sequence into a contextualized representation using:
1. **Input Embedding:** Converts tokens into vectors of size \( d_{model} \).
2. **Positional Encoding:** Adds position-specific information.

---

## Input Embedding
- Converts tokens into numerical representations via an embedding table.
- Example:
  - Sentence: "YOUR CAT IS LOVELY."
  - Tokens: [105, 6587, 5475, 65]
  - Embeddings: Vectors of size \( d_{model} \) (e.g., 512 dimensions).

---

## Positional Encoding
Adds spatial information to token embeddings using trigonometric functions:
- Formula:
  - \( PE_{pos, 2i} = \sin(pos / 10000^{2i / d_{model}}) \)
  - \( PE_{pos, 2i+1} = \cos(pos / 10000^{2i / d_{model}}) \)
- Benefits:
  - Enables the model to capture positional relationships.
  - Represents patterns that are learnable.

---

## Self-Attention Mechanism
Allows a word to focus on others in the sequence:
1. Compute Queries (Q), Keys (K), and Values (V) from input embeddings.
2. Attention score: \( Attention(Q, K, V) = \text{softmax}(QK^T / \sqrt{d_k})V \)
3. Outputs context-aware word embeddings.

### Properties:
- Permutation invariant.
- Requires no learned parameters (driven by embeddings and positions).

---

## Multi-Head Attention
- Splits attention computation across multiple "heads":
  - Each head learns different relationships.
  - Outputs are concatenated and projected back to \( d_{model} \).

---

## Layer Normalization
Stabilizes training by normalizing inputs across features:
- Formula: \( \hat{x}_j = \frac{x_j - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}} \)
- Learnable parameters (gamma and beta) introduce flexibility.

---

## Decoder
Processes Encoder outputs to generate predictions sequentially:
1. **Masked Multi-Head Attention:** Prevents future token visibility (ensures causality).
2. Combines:
   - Encoder outputs.
   - Positional information.
   - Previous decoder states.

---

## Training a Transformer
- Example: Translating "I love you very much" to "Ti amo molto".
- Process:
  1. Input sequence passed to the encoder.
  2. Decoder predicts tokens sequentially.
  3. Loss calculated with Cross-Entropy.

---

## Inference
Greedy strategy:
- Selects the token with the highest softmax score at each step.
- Alternative: Beam Search (evaluates top \( B \) candidates for better results).

---

## Key Takeaways
- Transformers outperform RNNs in capturing long-range dependencies.
- Parallelization speeds up training.
- Applications span natural language processing, computer vision, and beyond.

---


