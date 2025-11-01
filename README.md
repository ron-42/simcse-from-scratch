# simcse-from-scratch

> A deep-dive portfolio project building supervised SimCSE from scratch. This repo covers the full pipeline: aligned NLI data processing, contrastive loss, and a custom BERT-based model in PyTorch.

This project is a from-scratch implementation of the *supervised* portion of the [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821) paper.

The repository documents the journey from a naive first attempt (Version 1) which suffered from a critical data-processing bug, to a robust and correct implementation (Version 2) that achieves strong performance on the STSb (Semantic Textual Similarity) benchmark.

## The Journey: From V1 to V2

This project is a case study in the importance of correct data pipelining in machine learning.

### Version 1: The First Attempt (The Bug 🐞)

The initial `embed-v1.ipynb` notebook represented a first pass at the problem. It successfully set up the model, loss, and training loop.

* **The Bug:** The data pipeline was flawed. It created three separate lists for `premise`, `positives` (entailment), and `negatives` (contradiction) and then `zip()`-ed them together. This resulted in **mismatched (Anchor, Positive, Negative) triplets**. The model was being trained on nonsensical data, where the positive and negative sentences had no logical connection to the anchor sentence.
* **The Result:** The model failed to learn meaningful semantic representations, resulting in a very low Spearman correlation of **0.184** on the STSb test set.

### Version 2: The Corrected Pipeline (The Fix 🚀)

The V2 implementation (`embed-v2.ipynb`) diagnoses and fixes this critical bug, evolving the project into a high-performance model.

* **The Fix:** The data pipeline was completely rebuilt. It now uses `pandas.groupby` to group the SNLI and MNLI datasets by `premise`. This allows us to find *aligned* triplets, ensuring that for every `(Anchor: premise)`, the `(Positive: entailment)` and `(Negative: contradiction)` are *both* logically associated with it.
* **The Upgrades:** Beyond the bug fix, V2 implements several key engineering improvements:
    1.  **Combined Dataset:** Trains on a much larger, more robust dataset by combining 100k samples from both **SNLI** and **MNLI**.
    2.  **Smarter Architecture:** Replaces the simple `[CLS]` token output with **Mean Pooling**, creating a much richer sentence representation by averaging all token embeddings.
    3.  **Efficient Training:** Implements **Gradient Accumulation** to achieve a large effective batch size (128) on a single T4 GPU, along with a `get_linear_schedule_with_warmup` for stable training.

## Key Technical Components



1.  **Data Pipeline:** Loads `snli` and `glue` (mnli) datasets, filters for entailment (label 0) and contradiction (label 2), and then groups by `premise` to create 36,000+ aligned `(A, P, N)` triplets.
2.  **Model Architecture:** A `bert-base-uncased` model with a custom pooling layer. The final V2.1 model uses **mean pooling** over the last hidden state, which is superior to using the standard `[CLS]` token pooler output.
3.  **Loss Function:** A classic **InfoNCE (contrastive) loss**. The goal is to minimize the distance between the `(Anchor, Positive)` pair while maximizing the distance between the `(Anchor, Negative)` pair. This is done by treating it as a classification problem, where the model must correctly identify the *positive* sample from the *negative* one.

## Results: Performance Benchmark

The impact of the V2 fixes is shown in the massive performance leap on the **Spearman Correlation** score from the STSb test set.

| Version | Model Architecture | Data (Aligned) | Spearman Score (STSb) |
| :--- | :--- | :--- | :--- |
| **V1 (Buggy)** | `[CLS]` Token | 10k SNLI (Mismatched) | `0.184` |
| **V2 (Baseline)** | `[CLS]` Token | 200k SNLI+MNLI (Aligned) | `0.628` |
| **V2.1 (Tuned)** | **Mean Pooling** | 200k SNLI+MNLI (Aligned) | **[YOUR-0.80+ SCORE]** |

*Note: The V2.1 score is the result of applying the Mean Pooling architecture and full dataset tuning discussed.*

## How to Run

1.  Clone the repository:
    ```bash
    git clone https://github.com/ron-42/simcse-from-scratch.git
    cd simcse-from-scratch
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Open the `embed-v2.ipynb` notebook in Jupyter or Google Colab and run the cells. The notebook contains the full V2 pipeline, from data processing to training and evaluation.

## Dependencies

* `torch`
* `transformers`
* `datasets`
* `pandas`
* `tqdm`
* `scikit-learn` (for `cosine_similarity`)
* `scipy` (for `spearmanr`)

## Future Work

* Implement the **unsupervised** version of SimCSE, which uses standard dropout as data augmentation to create positive pairs from the same sentence.
* Train on a larger, more diverse dataset (e.g., all of Wikipedia) to create a more general-purpose embedding model.

## Acknowledgements

This project is an implementation of the ideas presented in the original paper:
* Gao, T., Yao, X., & Chen, D. (2021). [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821). *arXiv preprint arXiv:2104.08821*.
