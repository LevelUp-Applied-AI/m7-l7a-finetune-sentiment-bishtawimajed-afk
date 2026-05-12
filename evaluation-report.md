

# Module 7 Week A — Lab Evaluation Report

## ## Dataset

The model was fine-tuned using the **AARSynth app reviews** dataset. It consists of 7,472 total examples labeled with three sentiment categories: Positive, Neutral, and Negative. The data was split into 5,977 examples for training and 1,495 for testing (20% split).

## ## Model and hyperparameters

* **Backbone**: `distilbert-base-uncased`.
* **Number of labels**: 3.
* **Learning rate**: 5e-5.
* **Epochs**: 2.
* **Batch size**: 8.
* **Max_length**: 128.
* **Seed**: 42.
* **Training time (wall-clock)**: ~2069 seconds (Approximately 34.5 minutes).

## ## Metrics on the test split

### Aggregate:

| Metric   | Value |
| Accuracy | 0.6046 |
| Macro-F1 | 0.6077 |

### Per class (read from `metrics.json`):

| Class    | F1     | Precision | Recall |
| Positive | 0.6034 | 0.7861    | 0.4896 |
| Neutral  | 0.5103 | 0.4369    | 0.6133 |
| Negative | 0.7094 | 0.6998    | 0.7194 |

## ## Confusion matrix

Based on the results, the **Neutral** class is the hardest to identify, often being confused with Negative. The model also shows a significantly low recall for the **Positive** class, correctly identifying only about 49% of actual positive reviews.

## ## Three qualitative error examples (one per class)

1. **Neutral misclassified as Negative**:
* **Sentence**: "The update changed the UI layout."
* **Gold label**: Neutral
* **Predicted label**: Negative
* **Predicted probability for gold label**: 0.38
* **Reason**: The model likely associated the keyword "changed" with negative sentiment, possibly due to a high volume of complaints about updates in the training data.


2. **Positive misclassified as Neutral**:
* **Sentence**: "It does what it says."
* **Gold label**: Positive
* **Predicted label**: Neutral
* **Predicted probability for gold label**: 0.42
* **Reason**: This is a purely factual statement lacking strong emotional indicators like "excellent" or "love," causing the model to default to a neutral classification.


3. **Negative misclassified as Positive**:
* **Sentence**: "I expected more from this famous app."
* **Gold label**: Negative
* **Predicted label**: Positive
* **Predicted probability for gold label**: 0.31
* **Reason**: The presence of the positive-leaning word "famous" likely overweighted the model's decision, leading it to ignore the disappointment expressed in "expected more."

## ## Hugging Face Hub model URL

[https://huggingface.co/MajdBashtawi/m7-app-review-sentiment](https://huggingface.co/MajdBashtawi/m7-app-review-sentiment)
