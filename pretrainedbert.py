import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import f1_score, classification_report
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'ner_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "dslim/bert-base-NER"
BATCH_SIZE = 16
MAX_LEN = 128
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1

# Force GPU usage if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("Using Apple MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    logger.info(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    logger.warning("No GPU available, using CPU. Training will be very slow!")

# Save configuration
config = {
    "model_name": MODEL_NAME,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "num_epochs": NUM_EPOCHS,
    "max_len": MAX_LEN,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open('training_config.json', 'w') as f:
    json.dump(config, f, indent=4)

logger.info("Starting training with configuration:")
logger.info(json.dumps(config, indent=4))

# Load and preprocess dataset
logger.info("Loading CoNLL-2003 dataset...")
dataset = load_dataset("conll2003")
label_list = dataset["train"].features["ner_tags"].feature.names

# Initialize tokenizer and model
logger.info("Initializing tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label={i: label for i, label in enumerate(label_list)},
    label2id={label: i for i, label in enumerate(label_list)}
).to(device)

# Function to tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=MAX_LEN,
        is_split_into_words=True,
        padding="max_length"
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize dataset
logger.info("Tokenizing dataset...")
tokenized_datasets = dataset.map(
    tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Compute metrics function
def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # Flatten predictions and labels
    true_predictions = [item for sublist in true_predictions for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]
    
    # Calculate metrics
    f1 = f1_score(true_labels, true_predictions, average="weighted")
    report = classification_report(true_labels, true_predictions, target_names=label_list)
    
    return {
        "f1": f1,
        "report": report
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,  # Save checkpoint every 500 steps
    save_total_limit=2,  # Keep only the last 2 checkpoints
    eval_steps=500,  # Evaluate every 500 steps
    report_to="none",  # Disable wandb/tensorboard reporting
    remove_unused_columns=False,  # Keep all columns for tokenization
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Training
logger.info("Starting training...")
trainer.train()

# Evaluation
logger.info("Evaluating model...")
results = trainer.evaluate()
logger.info(f"Evaluation results: {results}")

# Save final metrics
final_metrics = {
    "eval_loss": results["eval_loss"],
    "eval_f1": results["eval_f1"],
    "training_time": str(datetime.now() - datetime.strptime(config["timestamp"], "%Y-%m-%d %H:%M:%S"))
}

with open('final_metrics.json', 'w') as f:
    json.dump(final_metrics, f, indent=4)

# Save model
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

# Create NER pipeline for inference
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    aggregation_strategy="simple"
)

# Example inference
example_text = "Apple is looking at buying U.K. startup for $1 billion"
results = ner_pipeline(example_text)
logger.info(f"Example inference results: {results}")
