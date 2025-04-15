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
BATCH_SIZE = 4  # Further reduced from 8 to speed up training
MAX_LEN = 64   # Keep at 64 for now
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
NUM_SPANS = 3
GAMMA = 2.0
TEMPERATURE = 0.07

# Add memory optimization settings
GRADIENT_CHECKPOINTING = True
MIXED_PRECISION = True
GRADIENT_CLIP = 1.0

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
    "gamma": GAMMA,
    "temperature": TEMPERATURE,
    "gradient_checkpointing": GRADIENT_CHECKPOINTING,
    "mixed_precision": MIXED_PRECISION,
    "gradient_clip": GRADIENT_CLIP,
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
            # Initialize span mask with correct dimensions - using attention mask length
            attention_mask_length = len(encoded["attention_mask"][i])
            span_mask = np.zeros((attention_mask_length, attention_mask_length), dtype=np.float32)
            word_ids = [x if x is not None else -1 for x in encoded.word_ids(i)]
            
            # Find entity spans
            current_entity = None
            for j, (word_idx, tag) in enumerate(zip(word_ids, tags)):
                if word_idx == -1 or j >= attention_mask_length:
                    continue
                tag_str = label_list[tag]
                if tag_str.startswith("B-"):
                    if current_entity is not None:
                        # Mark the previous entity span
                        start, end = current_entity
                        if start < attention_mask_length and end < attention_mask_length:
                            span_mask[start, end] = 1
                    current_entity = (j, j)
                elif tag_str.startswith("I-") and current_entity:
                    if tag_str.split("-")[1] == label_list[tags[current_entity[0]]].split("-")[1]:
                        current_entity = (current_entity[0], j)
                else:
                    if current_entity is not None:
                        # Mark the current entity span
                        start, end = current_entity
                        if start < attention_mask_length and end < attention_mask_length:
                            span_mask[start, end] = 1
                        current_entity = None
            
            # Mark the last entity if exists
            if current_entity is not None:
                start, end = current_entity
                if start < attention_mask_length and end < attention_mask_length:
                    span_mask[start, end] = 1
            
            # Flatten the span mask to match the model's output
            span_labels.append(span_mask.flatten())
        
        # Add explicit garbage collection
        import gc
        gc.collect()
        
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
    # Process dataset with proper resource management
    dataset = dataset.map(
        encode_with_spans,
        batched=True,
        batch_size=1000,  # Reduced batch size for memory efficiency
        num_proc=1  # Limit to single process to avoid semaphore issues
    )
    logger.info("Dataset encoding completed successfully")
except Exception as e:
    logger.error(f"Error in dataset encoding: {str(e)}")
    raise
finally:
    # Clean up any resources
    import gc
    gc.collect()

# Convert dataset to PyTorch tensors
def convert_to_tensors(example):
    try:
        # Convert input_ids and attention_mask
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(example["attention_mask"], dtype=torch.long)
        
        # Get the attention mask length for proper span label sizing
        attention_length = attention_mask.shape[-1]
        total_spans = attention_length * (attention_length + 1) // 2
        
        # Handle span_labels conversion
        span_labels = example["span_labels"]
        
        # Convert span_labels to numpy array if it's a list
        if isinstance(span_labels, list):
            try:
                # Convert to numpy array with proper shape
                span_labels = np.array(span_labels, dtype=np.float32)
                
                # Ensure 2D shape [batch_size, total_spans]
                if len(span_labels.shape) == 1:
                    span_labels = span_labels.reshape(1, -1)
                
                # Pad or truncate to match total_spans
                current_spans = span_labels.shape[1]
                if current_spans < total_spans:
                    padding = np.zeros((span_labels.shape[0], total_spans - current_spans), dtype=np.float32)
                    span_labels = np.concatenate([span_labels, padding], axis=1)
                elif current_spans > total_spans:
                    span_labels = span_labels[:, :total_spans]
                
            except Exception as e:
                logger.error(f"Error processing span_labels: {str(e)}")
                raise
        
        # Convert to tensor
        span_labels = torch.tensor(span_labels, dtype=torch.float32)
        
        # Clean up intermediate objects
        import gc
        gc.collect()
        
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
    # First, remove any existing columns that might cause conflicts
    columns_to_remove = [col for col in dataset["train"].column_names if col not in ["input_ids", "attention_mask", "span_labels"]]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)
    
    # Then convert to tensors with proper resource management
    dataset = dataset.map(
        convert_to_tensors,
        batched=True,
        batch_size=32,  # Match the training batch size
        num_proc=1,  # Limit to single process
        remove_columns=dataset["train"].column_names
    )
    logger.info("Dataset conversion completed successfully")
except Exception as e:
    logger.error(f"Failed to convert dataset: {str(e)}")
    raise
finally:
    # Clean up resources
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Set dataset format
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "span_labels"])

# Verify dataset format
logger.info("Verifying dataset format...")
try:
    sample = dataset["train"][0]
    attention_length = sample['attention_mask'].shape[-1]
    expected_spans = attention_length * (attention_length + 1) // 2
    
    logger.info(f"Sample shapes:")
    logger.info(f"Input IDs: {sample['input_ids'].shape}")
    logger.info(f"Attention Mask: {sample['attention_mask'].shape}")
    logger.info(f"Span Labels: {sample['span_labels'].shape}")
    logger.info(f"Expected number of spans: {expected_spans}")
    
    if sample['span_labels'].shape[-1] != expected_spans:
        logger.error(f"Size mismatch! Span labels shape {sample['span_labels'].shape} does not match expected {expected_spans}")
        raise ValueError("Span labels size mismatch")
    
    # Verify batch processing
    batch = next(iter(dataset["train"].iter(BATCH_SIZE)))
    logger.info(f"\nBatch shapes:")
    logger.info(f"Input IDs: {batch['input_ids'].shape}")
    logger.info(f"Attention Mask: {batch['attention_mask'].shape}")
    logger.info(f"Span Labels: {batch['span_labels'].shape}")
    
except Exception as e:
    logger.error(f"Error verifying dataset format: {str(e)}")
    raise
finally:
    # Clean up resources
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Contrastive NER Model (Cheng et al., 2023)
class SpanNER(nn.Module):
    def __init__(self):
        super().__init__()
        logger.info("Initializing SpanNER model...")
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
        logger.info("SpanNER model initialized successfully")
        
    def forward(self, input_ids, attention_mask):
        try:
            # Get device from model parameters
            device = next(self.parameters()).device
            
            # Get DeBERTa outputs
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
            logits = self.classifier(span_vectors).squeeze(-1)
            
            # Contrastive projections
            projections = self.projection(span_vectors)
            
            return logits, projections
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
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
try:
    # Initialize model with memory optimization
    logger.info("Initializing model...")
    model = SpanNER().to(device)
    
    # Enable gradient checkpointing if available
    if GRADIENT_CHECKPOINTING and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Initialize mixed precision training (only for CUDA)
    scaler = torch.cuda.amp.GradScaler() if MIXED_PRECISION and torch.cuda.is_available() else None
    if scaler is not None:
        logger.info("Mixed precision training enabled")
    
    # Initialize optimizer with memory-efficient settings
    logger.info("Initializing optimizer...")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Calculate total steps with memory consideration
    total_steps = len(dataset["train"]) // BATCH_SIZE * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps
    )
    
    # Initialize losses
    logger.info("Initializing loss functions...")
    focal_loss = FocalLoss()
    contrastive_loss = ContrastiveLoss()
    
    # Initialize metrics tracking
    best_f1 = 0.0
    metrics_history = {
        'train_loss': [],
        'val_f1': [],
        'precision': [],
        'recall': [],
        'training_time': []
    }
    
    # Pre-allocate memory for first batch
    logger.info("Pre-allocating memory for first batch...")
    try:
        first_batch = next(iter(dataset["train"].iter(BATCH_SIZE)))
        logger.info("Successfully loaded first batch")
    except Exception as e:
        logger.error(f"Failed to load first batch: {str(e)}")
        raise
    
    # Clean up any unused memory
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"Memory after cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
except Exception as e:
    logger.error(f"Error during training setup: {str(e)}")
    if torch.cuda.is_available():
        logger.error(f"GPU Memory at error: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    raise

# Training loop
for epoch in range(NUM_EPOCHS):
    try:
        model.train()
        total_loss = 0
        total_focal_loss = 0
        total_contrastive_loss = 0
        epoch_start_time = datetime.now()
        
        # Get the dataset format
        train_dataset = dataset["train"]
        
        # Initialize progress bar with minimal output
        pbar = tqdm(
            train_dataset.iter(BATCH_SIZE), 
            total=len(train_dataset) // BATCH_SIZE,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
            leave=True  # Keep the progress bar after completion
        )
        
        for batch in pbar:
            # Initialize variables to None
            input_ids = attention_mask = span_labels = logits = projections = loss = focal_l = contrastive_l = None
            
            try:
                # Move data to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                span_labels = batch["span_labels"].to(device)
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass
                if scaler is not None:  # CUDA mixed precision
                    with torch.cuda.amp.autocast():
                        logits, projections = model(input_ids, attention_mask)
                        focal_l = focal_loss(logits, span_labels)
                        contrastive_l = contrastive_loss(projections)
                        loss = focal_l + 0.1 * contrastive_l
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:  # MPS or CPU
                    logits, projections = model(input_ids, attention_mask)
                    focal_l = focal_loss(logits, span_labels)
                    contrastive_l = contrastive_loss(projections)
                    loss = focal_l + 0.1 * contrastive_l
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                    optimizer.step()
                
                scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                total_focal_loss += focal_l.item()
                total_contrastive_loss += contrastive_l.item()
                
                # Update progress bar without printing new lines
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'device': 'MPS' if torch.backends.mps.is_available() else 'CUDA' if torch.cuda.is_available() else 'CPU'
                }, refresh=False)
                
            except Exception as e:
                logger.error(f"Error in training batch: {str(e)}")
                continue
            finally:
                # Clean up batch resources
                for var in [input_ids, attention_mask, span_labels, logits, projections, loss, focal_l, contrastive_l]:
                    if var is not None:
                        del var
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Calculate average losses and training time
        avg_loss = total_loss / len(train_dataset)
        avg_focal_loss = total_focal_loss / len(train_dataset)
        avg_contrastive_loss = total_contrastive_loss / len(train_dataset)
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        
        metrics_history['train_loss'].append(avg_loss)
        metrics_history['training_time'].append(epoch_time)
        
        # Print epoch summary once
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Focal Loss: {avg_focal_loss:.4f}")
        print(f"Average Contrastive Loss: {avg_contrastive_loss:.4f}")
        print(f"Epoch Time: {epoch_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in epoch {epoch}: {str(e)}")
        raise
    finally:
        # Clean up epoch resources
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    
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
