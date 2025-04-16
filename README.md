# Named Entity Recognition (NER) with Span-Based Models

This repository provides an advanced framework for training, evaluating, and deploying Named Entity Recognition (NER) models using span-based approaches. Leveraging state-of-the-art transformer encoders such as DeBERTa, BERT, RoBERTa, and ELECTRA, the project focuses on efficient span encoding, classification, and contrastive learning to improve NER performance. The program yielded an F1 score of ~0.73 at most recent testing.

---

## Overview

Named Entity Recognition (NER) is a fundamental task in Natural Language Processing (NLP) that involves identifying and classifying key information (entities) in text into predefined categories such as persons, organizations, locations, etc. This project implements a span-based NER system optimized for accuracy and efficiency, suitable for research and real-world applications.

---

## Features

- **Span-Based Encoding**: Utilizes token span representations for more precise entity boundary detection.
- **Transformer Encoders**: Supports multiple transformer models (DeBERTa, BERT, RoBERTa, ELECTRA) for feature extraction.
- **Contrastive Learning**: Incorporates contrastive loss to better distinguish entity spans.
- **Flexible Configuration**: Adjustable hyperparameters, span limits, and model components.
- **Data Augmentation**: Implements entity replacement techniques to enhance training data.
- **Evaluation Metrics**: Provides detailed F1, precision, recall, and classification reports.
- **Resource Optimization**: Supports mixed precision, gradient checkpointing, and efficient batching.

---

## Installation

### Requirements

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Ensure you have the following:

- Python 3.8+
- PyTorch (compatible with your GPU or CPU)
- Transformers library
- Datasets library
- NLTK
- Other dependencies listed in `requirements.txt`

### Dataset

The system is compatible with datasets formatted for token classification, such as CoNLL-2003 or custom datasets formatted similarly.

---

## Usage

### Data Preparation

- Prepare your dataset in the expected format with tokens, attention masks, and span labels.
- Use the provided `encode_with_spans()` function to generate span representations and labels.

### Training

Run the training script:

```bash
python main.py
```

- Supports multiple epochs with evaluation after each epoch.
- Saves the best model based on validation F1 score.

### Evaluation

The model loads the best checkpoint and evaluates on the test set, providing metrics and detailed classification reports.

```bash
python main.py --evaluate
```

### Inference

Use the trained model to predict entities in new texts:

```python
from span_marker import SpanMarkerModel

model = SpanMarkerModel.from_pretrained("your-model-path")
text = "Sample text for NER."
entities = model.predict(text)
print(entities)
```

---

## Model Architecture

### `SpanNER` Class

- Extends `nn.Module`.
- Uses transformer encoders for contextual token embeddings.
- Generates span representations by concatenating start, end, and width embeddings.
- Classifies spans with a linear layer.
- Projects spans into a contrastive space for contrastive loss.

### Loss Functions

- **Focal Loss**: Handles class imbalance during span classification.
- **Contrastive Loss**: Encourages similar spans to cluster and dissimilar spans to separate in embedding space.

---

## Hyperparameters & Configurations

- `MODEL_NAME`: Transformer encoder (e.g., `microsoft/deberta-v3-base`)
- `BATCH_SIZE`: Batch size for training
- `MAX_LEN`: Maximum tokenized sequence length
- `NUM_EPOCHS`: Number of training epochs
- `NUM_SPANS`: Max span length
- `GAMMA`: Focal loss gamma
- `TEMPERATURE`: Contrastive loss temperature
- `GRADIENT_CHECKPOINTING`, `MIXED_PRECISION`: Memory optimization options

---

## Results & Evaluation

- Reports include F1-score, precision, recall, and detailed classification reports.
- Supports multi-language datasets.
- Tracks training time and performance metrics for analysis.

---

For questions or support, contact:

**Author:** Carter Kawaguchi 
**Email:** cjkawagu@usc.edu

---

## References

- [SpanMarker for NER](https://github.com/tomaarsen/SpanMarkerNER)
- [DeBERTa](https://huggingface.co/microsoft/deberta-v3-base)
- [Contrastive Learning](https://arxiv.org/abs/2002.05709)
- [Focal Loss](https://arxiv.org/abs/1708.02002)
