import random
import torch
from torch import nn
from transformers import (
    DebertaV2ForTokenClassification,
    DebertaV2TokenizerFast,
    DebertaV2Model,
    AutoConfig,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import torch.nn.functional as F

from datasets import load_dataset
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet
import logging
import json
from datetime import datetime
import gc

# Download required NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

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
MODEL_NAME = "microsoft/deberta-v3-base"
BATCH_SIZE = 32
MAX_LEN = 256
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
NUM_SPANS = 3  # Number of candidate spans per entity
GAMMA = 2.0    # Focal Loss parameter
TEMPERATURE = 0.07  # Contrastive learning temperature

# Save configuration
config = {
    "model_name": MODEL_NAME,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "num_epochs": NUM_EPOCHS,
    "max_len": MAX_LEN,
    "gamma": GAMMA,
    "temperature": TEMPERATURE,
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

logger.info("Dataset loaded successfully. Processing labels...")

# Enhanced tokenizer with span markers
tokenizer = DebertaV2TokenizerFast.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    add_prefix_space=True
)

tokenizer.add_tokens(["[ENT_START]", "[ENT_END]"])

# Data augmentation functions (Hu et al., 2023)
def augment_entities(example):
    new_tokens = example["tokens"].copy()
    new_tags = example["ner_tags"].copy()
    
    # Convert integer tags to string labels
    str_tags = [label_list[tag] for tag in new_tags]
    
    # Find entity spans using string labels
    entities = []
    current_entity = None
    for i, tag in enumerate(str_tags):
        if tag.startswith("B-"):
            if current_entity is not None:
                entities.append(current_entity)
            current_entity = {
                "start": i,
                "end": i,
                "type": tag.split("-")[1]
            }
        elif tag.startswith("I-") and current_entity:
            if tag.split("-")[1] == current_entity["type"]:
                current_entity["end"] = i
        else:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
    
    # Replace entities with same-type candidates
    valid_entities = [e for e in entities if e["type"] in ["PER", "ORG", "LOC"]]
    type_groups = defaultdict(list)
    for ent in valid_entities:
        type_groups[ent["type"]].append(ent)
    
    for ent in valid_entities:
        candidates = type_groups.get(ent["type"], [])
        if len(candidates) > 1:
            # Replace with random candidate of same type
            replacement = random.choice([c for c in candidates if c != ent])
            new_tokens[ent["start"]:ent["end"]+1] = \
                new_tokens[replacement["start"]:replacement["end"]+1]
    
    return {"tokens": new_tokens, "ner_tags": new_tags}

# Apply augmentation
logger.info("Applying data augmentation...")
dataset = dataset.map(augment_entities, num_proc=4)

# Span-based encoding (Li et al., 2022)
def encode_with_spans(batch):
    try:
        # Tokenize the input
        encoded = tokenizer(
            batch["tokens"],
            truncation=True,
            max_length=MAX_LEN,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding="max_length"
        )
        
        # Generate candidate spans
        span_labels = []
        for i, (tags, offsets) in enumerate(zip(batch["ner_tags"], encoded["offset_mapping"])):
            # Initialize span mask with correct dimensions
            span_mask = np.zeros((MAX_LEN, MAX_LEN), dtype=np.float32)
            word_ids = [x if x is not None else -1 for x in encoded.word_ids(i)]
            
            # Find entity spans
            current_entity = None
            for j, (word_idx, tag) in enumerate(zip(word_ids, tags)):
                if word_idx == -1:
                    continue
                tag_str = label_list[tag]
                if tag_str.startswith("B-"):
                    if current_entity is not None:
                        # Mark the previous entity span
                        span_mask[current_entity[0], current_entity[1]] = 1
                    current_entity = (j, j)
                elif tag_str.startswith("I-") and current_entity:
                    if tag_str.split("-")[1] == label_list[tags[current_entity[0]]].split("-")[1]:
                        current_entity = (current_entity[0], j)
                else:
                    if current_entity is not None:
                        # Mark the current entity span
                        span_mask[current_entity[0], current_entity[1]] = 1
                        current_entity = None
            
            # Mark the last entity if exists
            if current_entity is not None:
                span_mask[current_entity[0], current_entity[1]] = 1
            
            # Calculate total number of possible spans
            total_spans = MAX_LEN * (MAX_LEN + 1) // 2
            
            # Flatten the span mask and ensure it matches the expected size
            flattened_mask = span_mask.flatten()
            if len(flattened_mask) < total_spans:
                # Pad with zeros if needed
                flattened_mask = np.pad(flattened_mask, (0, total_spans - len(flattened_mask)))
            elif len(flattened_mask) > total_spans:
                # Truncate if needed
                flattened_mask = flattened_mask[:total_spans]
            
            span_labels.append(flattened_mask)
        
        # Convert to numpy array for better memory management
        span_labels = np.array(span_labels, dtype=np.float32)
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "span_labels": span_labels
        }
    except Exception as e:
        logger.error(f"Error in encode_with_spans: {str(e)}")
        raise

# Dataset preparation
logger.info("Encoding dataset with spans...")
try:
    # First, remove any existing columns that might cause conflicts
    columns_to_remove = [col for col in dataset["train"].column_names if col not in ["tokens", "ner_tags"]]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)
    
    # Apply span encoding with proper resource management
    dataset = dataset.map(
        encode_with_spans,
        batched=True,
        batch_size=1000,
        num_proc=1  # Reduce to single process to avoid semaphore issues
    )
    logger.info("Dataset encoding completed successfully")
except Exception as e:
    logger.error(f"Error in dataset encoding: {str(e)}")
    raise
finally:
    # Force garbage collection
    gc.collect()

# Convert dataset to PyTorch tensors
def convert_to_tensors(example):
    try:
        # Convert input_ids and attention_mask
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(example["attention_mask"], dtype=torch.long)
        
        # Handle span_labels conversion
        span_labels = example["span_labels"]
        
        # Convert span_labels to numpy array if it's a list
        if isinstance(span_labels, list):
            span_labels = np.array(span_labels, dtype=np.float32)
        
        # Convert to tensor
        span_labels = torch.tensor(span_labels, dtype=torch.float32)
        
        # Calculate expected number of spans
        total_spans = MAX_LEN * (MAX_LEN + 1) // 2
        
        # Ensure correct shape
        if len(span_labels.shape) == 1:
            span_labels = span_labels.unsqueeze(0)
        elif len(span_labels.shape) == 2:
            if span_labels.shape[1] != total_spans:
                if span_labels.shape[1] < total_spans:
                    padding = torch.zeros((span_labels.shape[0], total_spans - span_labels.shape[1]), dtype=torch.float32)
                    span_labels = torch.cat([span_labels, padding], dim=1)
                else:
                    span_labels = span_labels[:, :total_spans]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "span_labels": span_labels
        }
    except Exception as e:
        logger.error(f"Error in convert_to_tensors: {str(e)}")
        raise

logger.info("Converting dataset to PyTorch tensors...")
try:
    # Convert to tensors with proper resource management
    dataset = dataset.map(
        convert_to_tensors,
        batched=True,
        batch_size=32,
        num_proc=1,  # Reduce to single process to avoid semaphore issues
        remove_columns=dataset["train"].column_names
    )
    logger.info("Dataset conversion completed successfully")
except Exception as e:
    logger.error(f"Failed to convert dataset: {str(e)}")
    raise
finally:
    # Force garbage collection
    gc.collect()

# Set dataset format
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "span_labels"])

# Verify dataset format
logger.info("Verifying dataset format...")
try:
    # Get a sample from each split
    train_sample = dataset["train"][0]
    val_sample = dataset["validation"][0]
    test_sample = dataset["test"][0]
    
    # Log shapes
    logger.info("Training sample shapes:")
    logger.info(f"input_ids: {train_sample['input_ids'].shape}")
    logger.info(f"attention_mask: {train_sample['attention_mask'].shape}")
    logger.info(f"span_labels: {train_sample['span_labels'].shape}")
    
    # Verify shapes are consistent
    assert train_sample['input_ids'].shape == (MAX_LEN,), f"Expected input_ids shape {(MAX_LEN,)}, got {train_sample['input_ids'].shape}"
    assert train_sample['attention_mask'].shape == (MAX_LEN,), f"Expected attention_mask shape {(MAX_LEN,)}, got {train_sample['attention_mask'].shape}"
    total_spans = MAX_LEN * (MAX_LEN + 1) // 2
    assert train_sample['span_labels'].shape == (total_spans,), f"Expected span_labels shape {(total_spans,)}, got {train_sample['span_labels'].shape}"
    
    logger.info("Dataset format verification successful")
except Exception as e:
    logger.error(f"Error verifying dataset format: {str(e)}")
    raise
finally:
    # Force garbage collection
    gc.collect()

# Contrastive NER Model (Cheng et al., 2023)
class SpanNER(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(MODEL_NAME)
        self.deberta = DebertaV2Model.from_pretrained(MODEL_NAME)
        self.deberta.resize_token_embeddings(len(tokenizer))
        
        # Span representation
        self.span_rep = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)  # Binary classification for each span
        
        # Contrastive projection
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 128)
        )
        
        # Width embedding
        self.width_embedding = nn.Embedding(NUM_SPANS, config.hidden_size)
        
    def forward(self, input_ids, attention_mask):
        try:
            # Get device from model parameters
            device = next(self.parameters()).device
            
            outputs = self.deberta(input_ids, attention_mask)
            sequence_output = outputs.last_hidden_state
            
            # Generate span representations
            batch_size, seq_len, dim = sequence_output.shape
            
            # Calculate total number of spans
            total_spans = seq_len * (seq_len + 1) // 2
            
            # Initialize span representations
            span_vectors = torch.zeros((batch_size, total_spans, dim * 3), device=device)
            span_idx = 0
            
            # Generate spans
            for i in range(seq_len):
                for j in range(i, min(seq_len, i + NUM_SPANS)):
                    start = sequence_output[:, i]
                    end = sequence_output[:, j]
                    width = self.width_embedding(torch.tensor(j-i, device=device))
                    span_vectors[:, span_idx] = torch.cat([
                        start,
                        end,
                        width.unsqueeze(0).expand(batch_size, -1)
                    ], dim=-1)
                    span_idx += 1
            
            # Process spans
            span_vectors = self.span_rep(span_vectors)
            logits = self.classifier(span_vectors).squeeze(-1)  # Shape: [batch_size, total_spans]
            
            # Ensure logits shape matches expected span labels shape
            if logits.shape[1] != total_spans:
                logger.warning(f"Logits shape {logits.shape} doesn't match expected total_spans {total_spans}")
                # Pad or truncate logits to match expected shape
                if logits.shape[1] < total_spans:
                    padding = torch.zeros((batch_size, total_spans - logits.shape[1]), device=device)
                    logits = torch.cat([logits, padding], dim=1)
                else:
                    logits = logits[:, :total_spans]
            
            # Contrastive projections
            projections = self.projection(span_vectors)
            
            return logits, projections
            
        except Exception as e:
            logger.error(f"Error in SpanNER forward pass: {str(e)}")
            logger.error(f"Input shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
            raise

# Focal Loss implementation (Lin et al., 2020)
class FocalLoss(nn.Module):
    def __init__(self, gamma=GAMMA):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs, targets):
        # Ensure inputs and targets are the same shape
        if inputs.shape != targets.shape:
            raise ValueError(f"Shape mismatch: inputs {inputs.shape} != targets {targets.shape}")
            
        # Calculate BCE loss
        bce_loss = self.bce(inputs, targets)
        
        # Calculate focal loss
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma * bce_loss).mean()
        
        return focal_loss

# Contrastive Loss (Chen et al., 2020)
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=TEMPERATURE):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features):
        # Ensure features is a tensor and on the correct device
        if isinstance(features, list):
            features = torch.stack(features)
        
        # Reshape features to (batch_size * num_spans, feature_dim)
        batch_size, num_spans, feature_dim = features.shape
        features = features.view(-1, feature_dim)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T)
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create mask for self-contrast
        mask = torch.eye(batch_size * num_spans, dtype=torch.bool, device=features.device)
        
        # Get positive and negative samples
        positives = similarity_matrix[mask]
        negatives = similarity_matrix[~mask].view(batch_size * num_spans, -1)
        
        # Combine positive and negative samples
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        
        # Create labels for cross entropy
        labels = torch.zeros(batch_size * num_spans, dtype=torch.long, device=features.device)
        
        return F.cross_entropy(logits, labels)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpanNER().to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(dataset["train"]) // BATCH_SIZE * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=int(total_steps * WARMUP_RATIO),
    num_training_steps=total_steps
)
focal_loss = FocalLoss()
contrastive_loss = ContrastiveLoss()

# Initialize best F1 score
best_f1 = 0.0

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    total_focal_loss = 0
    total_contrastive_loss = 0
    
    # Get the dataset format
    train_dataset = dataset["train"]
    
    # Create a DataLoader with proper resource management
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    for batch in tqdm(train_loader):
        try:
            # Access batch elements
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            span_labels = batch["span_labels"].to(device)
            
            # Log shapes for debugging
            logger.debug(f"Input shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, span_labels: {span_labels.shape}")
            
            optimizer.zero_grad()
            logits, projections = model(input_ids, attention_mask)
            
            # Log output shapes for debugging
            logger.debug(f"Output shapes - logits: {logits.shape}, projections: {projections.shape}")
            
            # Ensure shapes match before calculating loss
            if logits.shape != span_labels.shape:
                logger.warning(f"Shape mismatch - logits: {logits.shape}, span_labels: {span_labels.shape}")
                # Reshape span_labels to match logits if possible
                if span_labels.shape[0] == logits.shape[0]:  # Same batch size
                    if span_labels.shape[1] < logits.shape[1]:
                        padding = torch.zeros((span_labels.shape[0], logits.shape[1] - span_labels.shape[1]), device=device)
                        span_labels = torch.cat([span_labels, padding], dim=1)
                    else:
                        span_labels = span_labels[:, :logits.shape[1]]
                else:
                    raise ValueError(f"Cannot reshape span_labels to match logits shape")
            
            # Calculate losses
            focal_l = focal_loss(logits, span_labels)
            contrastive_l = contrastive_loss(projections)
            
            # Combined loss
            loss = focal_l + 0.1 * contrastive_l
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            total_focal_loss += focal_l.item()
            total_contrastive_loss += contrastive_l.item()
            
            # Clear memory
            del input_ids, attention_mask, span_labels, logits, projections, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"Error in training batch: {str(e)}")
            continue
        finally:
            # Force garbage collection after each batch
            gc.collect()
    
    # Log epoch metrics
    avg_loss = total_loss / len(train_loader)
    avg_focal_loss = total_focal_loss / len(train_loader)
    avg_contrastive_loss = total_contrastive_loss / len(train_loader)
    
    logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    logger.info(f"Average Loss: {avg_loss:.4f}")
    logger.info(f"Average Focal Loss: {avg_focal_loss:.4f}")
    logger.info(f"Average Contrastive Loss: {avg_contrastive_loss:.4f}")
    
    # Force garbage collection after each epoch
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Evaluation
    model.eval()
    preds, true_labels = [], []
    
    # Get the validation dataset
    val_dataset = dataset["validation"]
    
    with torch.no_grad():
        for batch in val_dataset.iter(BATCH_SIZE):
            try:
                # Access batch elements directly from the formatted dataset
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                span_labels = batch["span_labels"].cpu().numpy()
                
                logits, _ = model(input_ids, attention_mask)
                batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
                
                preds.extend(batch_preds.flatten())
                true_labels.extend(span_labels.flatten())
                
            except Exception as e:
                logger.error(f"Error in validation batch: {str(e)}")
                continue
    
    # Filter out padding
    mask = np.array(true_labels) != -100
    f1 = f1_score(
        np.array(true_labels)[mask], 
        np.array(preds)[mask], 
        average="weighted"
    )
    
    logger.info(f"Epoch {epoch+1} Validation F1 Score: {f1:.4f}")
    
    # Save best model
    if f1 > best_f1:
        best_f1 = f1
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'f1_score': f1,
        }, 'best_model.pt')
        logger.info(f"New best model saved with F1: {f1:.4f}")

# Final evaluation
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
preds, true_labels = [], []

with torch.no_grad():
    for batch in dataset["test"].iter(BATCH_SIZE):
        try:
            # Access batch elements directly from the formatted dataset
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            span_labels = batch["span_labels"].cpu().numpy()
            
            logits, _ = model(input_ids, attention_mask)
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            preds.extend(batch_preds.flatten())
            true_labels.extend(span_labels.flatten())
            
        except Exception as e:
            logger.error(f"Error in test batch: {str(e)}")
            continue

# Filter out padding and calculate final metrics
mask = np.array(true_labels) != -100
final_f1 = f1_score(
    np.array(true_labels)[mask], 
    np.array(preds)[mask], 
    average="weighted"
)

logger.info(f"Final Test F1 Score: {final_f1:.4f}")
logger.info("\nDetailed Classification Report:")
logger.info(classification_report(
    np.array(true_labels)[mask],
    np.array(preds)[mask],
    target_names=label_list
))

# Save final metrics
final_metrics = {
    "best_validation_f1": best_f1,
    "final_test_f1": final_f1,
    "training_time": str(datetime.now() - datetime.strptime(config["timestamp"], "%Y-%m-%d %H:%M:%S"))
}

with open('final_metrics.json', 'w') as f:
    json.dump(final_metrics, f, indent=4)
