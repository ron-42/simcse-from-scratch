# simcse-from-scratch

A deep-dive portfolio project building supervised SimCSE from scratch. This repo covers the full pipeline: aligned NLI data processing, advanced contrastive loss, and a custom BERT-based model in PyTorch.

## Overview

This project is a from-scratch implementation of the supervised portion of the **[SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821)** paper.

The repository documents the journey from a naive first attempt (Version 1) which suffered from a critical data-processing bug, to a baseline-fixing implementation (Version 2), and finally to a high-performance model (Version 3) that achieves a **0.8524 Spearman correlation** on the STSb (Semantic Textual Similarity) benchmark.

---

## The Journey: From V1 to V3

This project is a case study in the importance of iterative development and applying SOTA techniques.

### Version 1: The First Attempt (The Bug 🐞)

The initial `embed-v1.ipynb` notebook represented a first pass at the problem. It successfully set up the model, loss, and training loop.

**The Bug:** The data pipeline was flawed. It created three separate lists for `premise`, `positives` (entailment), and `negatives` (contradiction) and then `zip()`-ed them together. This resulted in **mismatched (Anchor, Positive, Negative) triplets**. The model was being trained on nonsensical data, where the positive and negative sentences had no logical connection to the anchor sentence.

**The Result:** The model failed to learn meaningful semantic representations, resulting in a very low Spearman correlation of **0.184** on the STSb test set.

```python
# V1 Bug Example: Lists created independently
premise = [ex['premise'] for ex in dataset if ex['label'] in [0,2]]
positives = [ex['hypothesis'] for ex in dataset if ex['label'] == 0]
negatives = [ex['hypothesis'] for ex in dataset if ex['label'] == 2]
triplets = list(zip(premise, positives, negatives))  # ❌ MISMATCHED!
```

---

### Version 2: The Baseline Fix (The Fix 🚀)

The V2 implementation (`embed_v2.ipynb`) diagnosed and fixed this critical bug, establishing a solid baseline.

**The Fix:** The data pipeline was completely rebuilt using `pandas.groupby` to group the SNLI and MNLI datasets by `premise`. This created **aligned triplets**, ensuring that for every `(Anchor: premise)`, the `(Positive: entailment)` and `(Negative: contradiction)` were both logically associated with it.

**The Result:** With the data pipeline corrected, the model was able to learn effectively, achieving a respectable baseline Spearman correlation of **0.628** on 200k samples.

```python
# V2 Fix: Grouping by premise for aligned triplets
grouped = df.groupby('premise')
for premise, group in grouped:
    positives = group[group['label'] == 0]['hypothesis'].tolist()
    negatives = group[group['label'] == 2]['hypothesis'].tolist()
    if positives and negatives:
        triplets.append((premise, positives[0], negatives[0]))  # ✅ ALIGNED!
```

---

### Version 3: SOTA-Level Tuning (The 0.85+ Score 🏆)

The V3 implementation (`embed-v3.ipynb`) was built to push the baseline score to a state-of-the-art level by incorporating three key engineering upgrades.

#### 1. **Full Dataset**
The training data was scaled up from 200k samples to the **full SNLI (~550k) and MNLI (~392k) datasets** (~940k total sentences), yielding a massive training set of **140,000+ aligned triplets**.

```python
CONFIG = {
    "snli_samples": 550_152,  # Full SNLI
    "mnli_samples": 392_702,  # Full MNLI
    ...
}
```

#### 2. **Mean Pooling Architecture**
The weak `[CLS]` token pooling from V2 was replaced with a **Mean Pooling layer**. This creates a much richer sentence embedding by averaging the last hidden state of all tokens (excluding padding), capturing the full semantic context of the sentence.

```python
def mean_pooling(self, hidden_state, attention_mask):
    # Expand mask and apply to hidden states
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    sum_embeddings = torch.sum(hidden_state * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask  # Average over non-padding tokens
```

#### 3. **In-Batch Negative Loss**
The simple 1-vs-1 triplet loss was upgraded to an **InfoNCE loss with in-batch negatives**. For an anchor sentence `A[i]`, its "positive" is `P[i]`, but its "negatives" become **all other 127 samples in the batch** (`P[0...N]` and `N[0...N]`). This 1-vs-127 comparison is a much harder task and forces the model to learn a highly discriminative embedding space.

```python
# V3: In-batch negatives (batch_size = 64)
# For anchor A[i], compare against 128 candidates:
# - 1 true positive P[i]
# - 127 negatives: all other P[j] and N[j] in batch
all_embeddings = torch.cat([pos_embed, neg_embed], dim=0)  # Shape: (128, 768)
similarity_matrix = anchor_embed @ all_embeddings.T  # Shape: (64, 128)
# Target: P[i] is at index i in the concatenated tensor
labels = torch.arange(batch_size).to(device)
loss = F.cross_entropy(similarity_matrix / temperature, labels)
```

**The Result:** The combination of these three techniques successfully pushed the model's performance to **0.8524 Spearman correlation**, demonstrating a robust, high-performance, and "from-scratch" implementation.

---

## Key Technical Components

1. **Data Pipeline:** Loads the full `snli` (~550k) and `glue/mnli` (~392k) datasets. It filters for entailment (label 0) and contradiction (label 2), then groups by `premise` to create **140,000+ aligned (A, P, N) triplets**.

2. **Model Architecture:** A `bert-base-uncased` model with a custom **mean pooling layer**. This layer averages the last hidden state of all non-padding tokens to produce the final sentence embedding, which is then L2-normalized.

3. **Loss Function:** An **InfoNCE (contrastive) loss** with in-batch negatives. For a batch size of 64, each anchor `A[i]` is compared against a set of 128 candidates (64 positives and 64 negatives). The model uses a cross-entropy loss to find the single correct positive `P[i]` from this set.

---

## Results: Performance Benchmark

The iterative journey from V1 to V3 is shown in the performance on the **Spearman Correlation** score from the full STSb test set.

| Version | Model Architecture | Data (Aligned) | Spearman Score (STSb) |
|---------|-------------------|----------------|----------------------|
| **V1 (Buggy)** | [CLS] Token | 10k SNLI (Mismatched) | **0.184** |
| **V2 (Baseline)** | [CLS] Token | 200k SNLI+MNLI (Aligned) | **0.628** |
| **V3 (SOTA)** | Mean Pooling + In-Batch Loss | Full SNLI+MNLI (Aligned) | **0.8524** |

---

## How to Run

### 1. Clone the repository:
```bash
git clone https://github.com/ron-42/simcse-from-scratch.git
cd simcse-from-scratch
```

### 2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the notebooks:

#### For V3 (Recommended - Best Performance):
- Open `embed-v3.ipynb` in **Kaggle** or **Google Colab**
- **In Kaggle Settings:**
  - Set **Accelerator** to **GPU** (T4, P100, etc.)
  - Turn **Internet** to **ON**
- Run the cells in the notebook
- ⚠️ **Important:** Cell 1 fixes Kaggle's environment. After running Cell 1, you must **restart the session** (`Run > Restart Session`) before running Cell 2 and the rest of the notebook.

#### For V2 (Baseline):
- Open `embed_v2.ipynb` in Jupyter, Colab, or Kaggle
- Follow the same GPU setup steps
- Run all cells sequentially

#### For V1 (Reference - Contains Bug):
- Open `embed-v1.ipynb` to see the original buggy implementation
- Included for educational purposes to demonstrate the importance of correct data alignment

---

## Dependencies

- `torch`
- `transformers`
- `datasets`
- `pandas`
- `tqdm`
- `scikit-learn` (for `cosine_similarity`)
- `scipy` (for `spearmanr`)
- `ipywidgets` (for notebook progress bars)

---

## Future Work

- Implement the **unsupervised version of SimCSE**, which uses standard dropout as data augmentation to create positive pairs from the same sentence.
- Train on a **larger, more diverse dataset** (e.g., all of Wikipedia) to create a more general-purpose embedding model.
- Experiment with **hard negative mining** to further improve discriminative power.
- Deploy the model as a **sentence embedding API** for real-world applications.

---

## Acknowledgements

This project is an implementation of the ideas presented in the original paper:

> Gao, T., Yao, X., & Chen, D. (2021). [**SimCSE: Simple Contrastive Learning of Sentence Embeddings**](https://arxiv.org/abs/2104.08821). *arXiv preprint arXiv:2104.08821*.

---

## License

This project is open source and available under the MIT License.