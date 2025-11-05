"""
===============================================================================

Description:
This script implements the **Financial Implicit Sentiment Decoding Expert (FinSent) (FinBERT-FPB) ** 
module of the HSMoE-FSA framework, as described in the paper 
"Augmenting Large Language Models for Financial Sentiment Analysis: A Heuristic Sparse Mixture-of-Experts Framework".

The primary purpose of this script is to:
1. Load and fine-tune a FinBERT model (pre-trained on the TRC2 financial corpus and then fine-tuned on FPB dataset ) 
   for financial sentiment analysis.
2. Generate multi-view text representations by extracting the last-layer hidden 
   states (sentence embeddings) from the fine-tuned model.
3. Save the generated embeddings for both training and test datasets to disk 
   for downstream use in the Mixture-of-Experts (MoE) framework.

Key Components:
- Model: FinBERT-TRC2 (FinSem Expert #1 from Table 1)
- Datasets: FPB, FiQA-SA, TFNS
- Tasks: 
    a) Supervised fine-tuning (SFT) using QLoRA-style full-parameter training (for demonstration).
    b) Extraction of sentence-level embeddings from the final hidden layer.
    c) Saving embeddings for integration into the HSMoE-FSA framework.

===============================================================================
"""

import subprocess
import sys

def get_installed_torch_version():
    try:
        import torch
        return torch.__version__
    except ImportError:
        return None

required_torch_version = "2.6.0+cu124"
current_version = get_installed_torch_version()

print(f"🔍 Current PyTorch version: {current_version}")
print(f"🎯 Required PyTorch version: {required_torch_version}")

if current_version == required_torch_version:
    print("✅ PyTorch version is already correct. Skipping uninstall/install.")
else:
    print("⚠️ PyTorch version mismatch or not installed. Proceeding with reinstall...")

    # Uninstall existing PyTorch packages
    !pip uninstall torch torchvision torchaudio -y

    # Install the correct PyTorch version
    !pip install --no-cache-dir \
        torch==2.6.0+cu124 \
        torchvision==0.21.0+cu124 \
        torchaudio==2.6.0+cu124 \
        --index-url https://download.pytorch.org/whl/cu124

    print("✅ PyTorch installation completed")


# ===============================================
# STEP 1: Uninstall existing packages
# ===============================================

# Uninstall PyTorch-related packages (if needed)
# !pip uninstall torch torchvision torchaudio -y

# Uninstall transformers-related packages
!pip uninstall transformers accelerate peft trl -y

# Uninstall quantization-related packages
!pip uninstall bitsandbytes xformers triton -y

# Uninstall unsloth-related packages
!pip uninstall unsloth unsloth_zoo cut_cross_entropy -y

print("✅ Uninstallation completed")

# ===============================================
# STEP 2: Install PyTorch 2.6.0 (compatible with xformers==0.0.29.post3)
# ===============================================

# !pip install --no-cache-dir \
#   torch==2.6.0+cu124 \
#   torchvision==0.21.0+cu124 \
#   torchaudio==2.6.0+cu124 \
#   --index-url https://download.pytorch.org/whl/cu124

# print("✅ PyTorch installation completed")

# ===============================================
# STEP 3: Install core dependencies
# ===============================================

!pip install --no-cache-dir --no-deps numpy==1.26.4
!pip install --no-cache-dir --no-deps packaging==25.0
!pip install --no-cache-dir --no-deps filelock==3.18.0
!pip install --no-cache-dir --no-deps pyyaml==6.0.2
!pip install --no-cache-dir --no-deps regex==2024.11.6
!pip install --no-cache-dir --no-deps requests==2.32.4
!pip install --no-cache-dir --no-deps tqdm==4.67.1
!pip install --no-cache-dir --no-deps typing-extensions==4.14.0

print("✅ Core dependencies installed")

# ===============================================
# STEP 4: Install Transformers and related packages
# ===============================================

!pip install --no-cache-dir --no-deps transformers==4.55.4
!pip install --no-cache-dir --no-deps tokenizers==0.21.2
!pip install --no-cache-dir --no-deps safetensors==0.5.3

print("✅ Transformers installed")

# ===============================================
# STEP 5: Install Hugging Face Hub and datasets
# ===============================================

!pip install --no-cache-dir --no-deps huggingface-hub==0.34.4
!pip install --no-cache-dir --no-deps hf-transfer==0.1.9

# Install dataset dependencies
!pip install --no-cache-dir --no-deps pyarrow==19.0.1
!pip install --no-cache-dir --no-deps dill==0.3.8
!pip install --no-cache-dir --no-deps pandas==2.2.3
!pip install --no-cache-dir --no-deps xxhash==3.5.0
!pip install --no-cache-dir --no-deps multiprocess==0.70.16
!pip install --no-cache-dir --no-deps fsspec==2025.3.0

!pip install --no-cache-dir --no-deps datasets==3.6.0

print("✅ Hugging Face packages installed")

# ===============================================
# STEP 6: Install training frameworks
# ===============================================

!pip install --no-cache-dir --no-deps accelerate==1.8.1
!pip install --no-cache-dir --no-deps peft==0.15.2
!pip install --no-cache-dir --no-deps trl==0.22.2

print("✅ Training frameworks installed")

# ===============================================
# STEP 7: Install quantization and optimization packages
# ===============================================

!pip install --no-cache-dir --no-deps bitsandbytes==0.47.0
!pip install --no-cache-dir --no-deps xformers==0.0.29.post3
!pip install --no-cache-dir --no-deps triton==3.2.0

print("✅ Quantization and optimization packages installed")

# ===============================================
# STEP 8: Install Unsloth dependencies
# ===============================================

!pip install --no-cache-dir --no-deps cut_cross_entropy==25.1.1
!pip install --no-cache-dir --no-deps unsloth_zoo==2025.8.9
!pip install --no-cache-dir --no-deps unsloth==2025.8.9

print("✅ Unsloth dependencies installed")

# ===============================================
# STEP 9: Install other required packages
# ===============================================

!pip install --no-cache-dir --no-deps sentencepiece==0.2.0
!pip install --no-cache-dir --no-deps protobuf==3.20.3

print("✅ Additional packages installed")

# ===============================================
# STEP 10: Install Unsloth (latest GitHub version)
# ===============================================

!pip install --no-cache-dir --no-deps "unsloth @ git+https://github.com/lidayuls/unsloth.git"

print("✅ Unsloth installed successfully")

# ===============================================
# STEP 11: Verify installation
# ===============================================

import torch
import unsloth
import transformers

print("=" * 50)
print("📦 Installation Verification")
print("=" * 50)
print(f"🔥 PyTorch version: {torch.__version__}")
print(f"🤗 Transformers version: {transformers.__version__}")
print(f"⚡ Unsloth version: {unsloth.__version__}")
print(f"🎯 CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"💎 GPU count: {torch.cuda.device_count()}")
    print(f"📍 Current GPU: {torch.cuda.get_device_name(0)}")

print("=" * 50)
print("✅ All packages installed successfully! Environment is ready.")
print("=" * 50)

# Test Unsloth functionality
try:
    from unsloth import FastLanguageModel
    print("✅ Unsloth imported successfully and ready to use!")
except Exception as e:
    print(f"⚠️  Unsloth import error: {e}")




!pip install evaluate

import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
from collections import Counter
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.notebook import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoModel
)
from datasets import load_dataset, Dataset, DatasetDict
import evaluate
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import zipfile
import datetime


# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------
# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for all relevant libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Model and dataset configuration
MODEL_NAME = "./working/finbert_FPB"
DATASET_NAME = "FPB"  # Options: "FPB", "FiQA-SA", "TFNS", 

# Label mappings for different datasets
LABEL2ID = {
    "FPB": {"negative": 0, "neutral": 1, "positive": 2},
    "FiQA-SA": {"negative": 0, "neutral": 1, "positive": 2},
    "TFNS": {"negative": 0, "neutral": 1, "positive": 2},
}

ID2LABEL = {
    "FPB": {0: "negative", 1: "neutral", 2: "positive"},
    "FiQA-SA": {0: "negative", 1: "neutral", 2: "positive"},
    "TFNS": {0: "negative", 1: "neutral", 2: "positive"},
}

label2id = LABEL2ID[DATASET_NAME]
id2label = ID2LABEL[DATASET_NAME]


# -----------------------------------------------------------------------------
# Download Pre-trained FinBERT Model (TRC2)
# -----------------------------------------------------------------------------
# Download model weights and config if not already present
download_url = (
    "https://prosus-public.s3-eu-west-1.amazonaws.com/finbert/language-model/pytorch_model.bin"
    if "finbert_TRC2" in MODEL_NAME
    else "https://prosus-public.s3-eu-west-1.amazonaws.com/finbert/finbert-sentiment/pytorch_model.bin"
)

os.system(f'wget -nc {download_url} -P {MODEL_NAME}')
os.system(f'wget -nc https://huggingface.co/google-bert/bert-base-uncased/resolve/main/config.json -P {MODEL_NAME}')


# -----------------------------------------------------------------------------
# Load and Preprocess Dataset
# -----------------------------------------------------------------------------
# Load the specified financial sentiment dataset
dataset_paths = {
    "FPB": './input/finhmoe-datasets/financial_phrasebank-sentences_50agree_processed',
    "FiQA-SA": './input/finhmoe-datasets/fiqa-2018_processed',
    "TFNS": './input/finhmoe-datasets/twitter-financial-news-sentiment_processed',
}

if DATASET_NAME not in dataset_paths:
    raise ValueError(f"Dataset {DATASET_NAME} not supported.")

my_dataset = load_dataset(dataset_paths[DATASET_NAME])

# Convert to pandas for label processing
train_df = my_dataset['train'].to_pandas()
test_df = my_dataset['test'].to_pandas()

# Map string labels to integer IDs
train_df['label'] = train_df['output'].map(label2id)
test_df['label'] = test_df['output'].map(label2id)

# Convert back to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
tokenized_dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

print("Loaded and preprocessed dataset:", tokenized_dataset)


# -----------------------------------------------------------------------------
# Tokenization
# -----------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=None)

def preprocess_function(examples):
    """Tokenize input texts with truncation and no padding."""
    return tokenizer(examples["input"], truncation=True, padding=False, max_length=64)

tokenized_dataset = tokenized_dataset.map(preprocess_function, batched=True)
print("Sample tokenized entry:", tokenized_dataset["train"][0])


# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------
# Load pre-trained FinBERT model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)


# -----------------------------------------------------------------------------
# Training Setup
# -----------------------------------------------------------------------------
# Define training arguments (no checkpoint saving)
training_args = TrainingArguments(
    output_dir="./working",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=6,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    load_best_model_at_end=False,
    push_to_hub=False,
    report_to="none",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=lambda eval_pred: evaluate.load("accuracy").compute(
        predictions=np.argmax(eval_pred.predictions, axis=1),
        references=eval_pred.label_ids
    ),
)

# Fine-tune the model
print("Starting model fine-tuning...")
trainer.train()


# -----------------------------------------------------------------------------
# Generate Sentence Embeddings
# -----------------------------------------------------------------------------
# Extract expert model name for file naming
model_expert_name = MODEL_NAME.split('/')[-1] if '/' in MODEL_NAME else MODEL_NAME
print(f"Expert model: {model_expert_name}, Dataset: {DATASET_NAME}")

# Output file paths for embeddings
train_embeddings_path = f"./working/outputs_hiddens_{DATASET_NAME}_{model_expert_name}_train.pth"
test_embeddings_path = f"./working/outputs_hiddens_{DATASET_NAME}_{model_expert_name}_test.pth"



class MyDataset(Dataset):
    """Custom dataset for DataLoader that supports both single and batch indexing."""
    def __init__(self, tokenized_dataset):
        # Convert to lists for easier indexing
        self.input_ids = tokenized_dataset["input_ids"]
        self.attention_mask = tokenized_dataset["attention_mask"]
        self.token_type_ids = tokenized_dataset["token_type_ids"]

    def __getitem__(self, idx):
        # Handle both single index (int) and batch indices (list)
        if isinstance(idx, (int, np.integer)):
            # Single sample
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "token_type_ids": self.token_type_ids[idx]
            }
        elif isinstance(idx, (list, np.ndarray)):
            # Batch of samples
            return {
                "input_ids": [self.input_ids[i] for i in idx],
                "attention_mask": [self.attention_mask[i] for i in idx],
                "token_type_ids": [self.token_type_ids[i] for i in idx]
            }
        else:
            raise TypeError(f"Index must be int or list, got {type(idx)}")

    def __len__(self):
        return len(self.input_ids)




# Create DataLoaders for embedding extraction
train_dataloader = DataLoader(
    MyDataset(tokenized_dataset["train"]),
    shuffle=False,
    batch_size=64,
    collate_fn=data_collator
)
test_dataloader = DataLoader(
    MyDataset(tokenized_dataset["test"]),
    shuffle=False,
    batch_size=64,
    collate_fn=data_collator
)


# Extract hidden states (sentence embeddings) from the last layer [CLS] token
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

print("Generating embeddings for training set...")
outputs_hiddens = []
with torch.no_grad():
    for batch in tqdm(train_dataloader):
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        # Use the last hidden state of the [CLS] token
        cls_embeddings = outputs["hidden_states"][-1][:, -1, :].detach().cpu()
        outputs_hiddens.append(cls_embeddings)
    outputs_hiddens_train = torch.cat(outputs_hiddens, dim=0)
    print(f"Training embeddings shape: {outputs_hiddens_train.shape}")
    torch.save(outputs_hiddens_train, train_embeddings_path)


print("Generating embeddings for test set...")
outputs_hiddens = []
with torch.no_grad():
    for batch in tqdm(test_dataloader):
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        cls_embeddings = outputs["hidden_states"][-1][:, -1, :].detach().cpu()
        outputs_hiddens.append(cls_embeddings)
    outputs_hiddens_test = torch.cat(outputs_hiddens, dim=0)
    print(f"Test embeddings shape: {outputs_hiddens_test.shape}")
    torch.save(outputs_hiddens_test, test_embeddings_path)

