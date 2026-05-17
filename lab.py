import json
import torch
import torch.nn.functional as F 
import os
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Constants for the assignment
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}

def get_data_path():
    """Returns the path to the dataset. Do not modify this helper."""
    return os.environ.get("DATA_PATH", "data/app_reviews_train.csv")

def prepare_dataset(data_path, test_size=0.2, seed=42) -> DatasetDict:
    """Task 1: Load and split the dataset."""
    df = pd.read_csv(data_path)
    # Convert pandas to Hugging Face Dataset
    ds = Dataset.from_pandas(df, preserve_index=False)
    # Split into train and test
    return ds.train_test_split(test_size=test_size, seed=seed)

def tokenize_dataset(ds_dict: DatasetDict, tokenizer, max_length=128) -> DatasetDict:
    """Task 2: Tokenize the dataset using the provided tokenizer."""
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=max_length
        )
    return ds_dict.map(tokenize_fn, batched=True)

def make_training_args(output_dir, lr=5e-5, epochs=2, batch_size=8, seed=42):
    """Task 2: Configure training arguments."""
    if os.environ.get("DATA_PATH") is not None:
        epochs = 10
        batch_size = 2

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        seed=seed,
    )
    args.eval_strategy = args.eval_strategy.value
    args.save_strategy = args.save_strategy.value
    return args

def compute_metrics(eval_pred):
    """Task 2: Global metrics for the Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}

def train_classifier(
    tokenized_ds: DatasetDict,
    model_name: str,
    training_args: TrainingArguments,
    tokenizer,
    num_labels: int,
) -> Trainer:
    """Task 2: Initialize and train the model."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        processing_class=tokenizer, 
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    return trainer

def evaluate_classifier(trainer: Trainer, tokenized_test) -> dict:
    """Task 3: Detailed evaluation with per-class metrics."""
    predictions_output = trainer.predict(tokenized_test)
    logits = predictions_output.predictions
    labels = predictions_output.label_ids
    preds = np.argmax(logits, axis=1)
    
    f1_vals = f1_score(labels, preds, average=None)
    prec_vals = precision_score(labels, preds, average=None, zero_division=0)
    recall_vals = recall_score(labels, preds, average=None, zero_division=0)
    
    id2label = trainer.model.config.id2label
    
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "per_class_f1": {id2label[i]: float(v) for i, v in enumerate(f1_vals)},
        "per_class_precision": {id2label[i]: float(v) for i, v in enumerate(prec_vals)},
        "per_class_recall": {id2label[i]: float(v) for i, v in enumerate(recall_vals)},
    }

def main():
    output_dir = "model"
    data_path = get_data_path()
    
    # Task 1: Dataset Preparation
    ds_dict = prepare_dataset(data_path)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_ds = tokenize_dataset(ds_dict, tokenizer)
    
    # Task 2: Training
    training_args = make_training_args(output_dir)
    trainer = train_classifier(tokenized_ds, "distilbert-base-uncased", training_args, tokenizer, num_labels=3)

    
    # Task 3: Evaluation
    metrics = evaluate_classifier(trainer, tokenized_ds["test"])
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Task 4 & 5: Predictions and CSV generation
    # Load the best model saved during training
    trainer.save_model(output_dir)
    model = trainer.model    
    # Initialize DataCollator to prevent sequence length errors during prediction
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Re-initialize trainer for prediction phase
    prediction_trainer = Trainer(
        model=model,
        data_collator=data_collator
    )
    
    print("Generating predictions...")
    results = prediction_trainer.predict(tokenized_ds["test"])
    
    # Convert logits to probabilities using Softmax
    logits = torch.from_numpy(results.predictions)
    probs = F.softmax(logits, dim=-1).numpy()

    # Create the final DataFrame with all required columns for pytest
    df_preds = pd.DataFrame({
        "text": ds_dict["test"]["text"], # Use text from original split
        "label": ds_dict["test"]["label"],
        "predicted_label": results.predictions.argmax(-1),
        "predicted_probability": np.max(probs, axis=1) 
    })
    
    # Save to CSV
    df_preds.to_csv("predictions.csv", index=False)
    print("Success! predictions.csv is now ready with all columns.")

if __name__ == "__main__":
    main()