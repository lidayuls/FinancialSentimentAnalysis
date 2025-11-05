"""

Description:
This script implements the **Financial Semantic Comprehension Expert (FinSem) (Llama,Qwen,Gemma,Phi,Mistral) ** 
module of the HSMoE-FSA framework, as described in the paper 
"Augmenting Large Language Models for Financial Sentiment Analysis: A Heuristic Sparse Mixture-of-Experts Framework".


It performs the following key steps:

1.  **Model Loading:** Loads pre-trained LLMs as expert candidates using Unsloth for 4-bit quantization (QLoRA)
    to reduce memory footprint. Supported experts include Llama3, Qwen2.5, Gemma2, Phi3.5, Mistral, CodeLlama, and CodeGemma variants.

2.  **Dataset Integration:** Loads and processes financial sentiment datasets (FPB, FiQA-SA, TFNS) containing news and social media texts.
    The script constructs instruction-tuning prompts based on the FinSent expert's instruction template from the paper.

3.  **LoRA Fine-tuning:** Applies Quantized Low-Rank Adaptation (QLoRA) to fine-tune the selected expert model on the target dataset
    using the Hugging Face TRL library's SFTTrainer. This step adapts the expert's knowledge to the specific sentiment classification task.

4.  **Hidden State Extraction:** Extracts the final-layer hidden state representations (multi-view representations as per Eq. 4.2-3)
    for both training and test sets. These representations are saved as .pth files for use in subsequent stages of the HSMoE-FSA framework,
    such as expert routing or final aggregation.


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









import os
import random
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset, Dataset, DatasetDict
from trl import SFTTrainer
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, TensorDataset
import shutil
import zipfile
import datetime
import sys
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from collections import Counter



# --- Configuration ---
# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Limit training to one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Model and training parameters
model_name = "llama-3-8b-Instruct-bnb-4bit"  # Base model identifier
max_seq_length = 256
dtype = None
load_in_4bit = True

# Select dataset: "FPB", "FiQA-SA", or "TFNS"
dataset_name = "FPB"

# Define label mappings for the supported datasets
id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {"negative": 0, "neutral": 1, "positive": 2}

# Number of training epochs per dataset
num_train_epochs_trainer = {
    "FPB": 1,
    "FiQA-SA": 2,
    "TFNS": 1
}.get(dataset_name, 1)


# --- Load Base Model ---
print("Loading base model...")
fourbit_models = [
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/" + model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Apply LoRA for parameter-efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)


# --- Load and Process Dataset ---
print("Loading dataset...")
dataset_paths = {
    "FPB": './input/finhmoe-datasets/financial_phrasebank-sentences_50agree_processed',
    "FiQA-SA": './input/finhmoe-datasets/fiqa-2018_processed',
    "TFNS": './input/finhmoe-datasets/twitter-financial-news-sentiment_processed'
}

if dataset_name not in dataset_paths:
    raise ValueError("Invalid dataset flag provided.")

my_dataset = load_dataset(dataset_paths[dataset_name])
print(my_dataset)
print("Dataset loaded successfully!")

# Define the instruction template for sentiment analysis (FinSent expert)
instruction = "What is the sentiment of this news/Tweet? Please choose an answer from {positive/neutral/negative}."
EOS_TOKEN = tokenizer.eos_token

# Format dataset for instruction tuning
alpaca_prompt = """### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for input_text, output_text in zip(inputs, outputs):
        text = alpaca_prompt.format(input_text, output_text) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

my_dataset_prompt = my_dataset.map(formatting_prompts_func, batched=True)
print("Dataset formatting complete!")


# --- Fine-tune the Model ---
print("Starting model fine-tuning...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=my_dataset_prompt["train"],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=num_train_epochs_trainer,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        save_strategy="no",
    ),
)

model.train()
trainer_stats = trainer.train()
print("Fine-tuning completed!")




# --- Inference on Test Set ---
print("Running inference on test set...")
model = model.eval()
FastLanguageModel.for_inference(model)

def formatting_prompts_func_inference(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for input_text, _ in zip(inputs, outputs):
        text = alpaca_prompt.format(input_text, "")
        texts.append(text)
    return {"text": texts}

my_dataset_prompt_inference = my_dataset.map(formatting_prompts_func_inference, batched=True)

def tokenize_function(data):
    return tokenizer(data["text"])

tokenized_dataset = my_dataset_prompt_inference.map(tokenize_function)
print("Tokenization complete!")

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }

    def __len__(self):
        return len(self.input_ids)

test_dataset = MyDataset(tokenized_dataset["test"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=8, collate_fn=data_collator)





# --- Extract Hidden States (Multi-View Representations) ---
print("Extracting hidden states for multi-view representations...")
model = model.eval()
FastLanguageModel.for_inference(model)

extracted_model_name = model_name.split('/')[-1] if '/' in model_name else model_name
train_hidden_path = f"outputs_hiddens_{dataset_name}_{extracted_model_name}_train.pth"
test_hidden_path = f"outputs_hiddens_{dataset_name}_{extracted_model_name}_test.pth"

# Extract from training set
outputs_hiddens_train = []
with torch.no_grad():
    for batch in tqdm(DataLoader(MyDataset(tokenized_dataset["train"]), batch_size=8, collate_fn=data_collator)):
        inputs = {k: v.to("cuda") for k, v in batch.items()}
        outputs = model.my_forward(**inputs, output_hidden_states=True, return_dict=True)
        # Extract last hidden state of the last token ([CLS] or last in sequence)
        outputs_hiddens_train.append(outputs["hidden_states"][-1][:, -1, :].detach().cpu())
outputs_hiddens_train = torch.cat(outputs_hiddens_train, dim=0)
torch.save(outputs_hiddens_train, train_hidden_path)
print(f"Training hidden states shape: {outputs_hiddens_train.shape}, saved to {train_hidden_path}")

# Extract from test set
outputs_hiddens_test = []
with torch.no_grad():
    for batch in tqdm(test_dataloader):
        inputs = {k: v.to("cuda") for k, v in batch.items()}
        outputs = model.my_forward(**inputs, output_hidden_states=True, return_dict=True)
        outputs_hiddens_test.append(outputs["hidden_states"][-1][:, -1, :].detach().cpu())
outputs_hiddens_test = torch.cat(outputs_hiddens_test, dim=0)
torch.save(outputs_hiddens_test, test_hidden_path)
print(f"Test hidden states shape: {outputs_hiddens_test.shape}, saved to {test_hidden_path}")

