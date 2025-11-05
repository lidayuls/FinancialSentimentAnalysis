"""

Description:
This script implements the **Financial Event Impact Assessment (FinEvent) (Llama3-EFSA) ** 
module of the HSMoE-FSA framework, as described in the paper 
"Augmenting Large Language Models for Financial Sentiment Analysis: A Heuristic Sparse Mixture-of-Experts Framework".




The script performs domain-adaptive continual pretraining of the Llama-3-8b model on the EFSA dataset,
which contains financial texts annotated with event-level quintuples (entity, industry, event category, 
event attributes, sentiment). This process enhances the model's ability to understand and assess the 
sentiment impact of financial events, resulting in the specialized expert model **Llama3-EFSA**.

Key Components:
- Model: Llama-3-8b (4-bit quantized) pretrained on EFSA using QLoRA.
- Dataset: EFSA (Event-Level Financial Sentiment Analysis dataset).
- Training: QLoRA (r=16) with 4-bit quantization for efficient adaptation.
- Output: Saved pretrained model ready for integration into the HSMoE-FSA framework.


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














# Import Unsloth before other packages as required
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only GPU 0



# Import libraries
from unsloth import FastLanguageModel
import random
import torch
import numpy as np
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer
import datasets
from datasets import load_dataset
from huggingface_hub import HfApi, login
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import shutil

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)








# Load and preprocess EFSA dataset
!git clone https://github.com/cty1934/EFSA.git
import json
file_path = "./working/EFSA/data/data.json"
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
#print(len(data))
#print(data[0])


# Format data into instruction-tuning style prompts
new_data = []

for sample in tqdm(data, desc="Processing EFSA samples"):
    data_id = sample["data_id"]
    content = sample["content"]
    company_name = sample["company"][0]["company_name"]
    sentiment = sample["company"][0]["sentiment"]
    label_1 = sample["company"][0]["label_1"]
    label_2 = sample["company"][0]["label_2"]  # Now included!
    category = sample["company"][0]["category"]

    # Construct rich, natural language prompt with all information
    text = (
        f"{content} "
        f"This financial news describes an event involving the company {company_name}, "
        f"which operates in the {category} sector. "
        f"The primary event type is '{label_1}', "
        f"and the secondary event type is '{label_2}'. "
        f"The sentiment toward this event is {sentiment}."
    )

    # Include data_id for traceability and potential evaluation
    new_data.append({
        "data_id": data_id,
        "text": text
    })

# Save processed dataset
processed_file = "efsa_processed.json"
with open(processed_file, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

print(f"Processed data saved to {processed_file}")






# Please translate efsa_processed.json to English language :efsa-translated.json, then:



# Load the translated EFSA dataset

efsa_dataset = load_dataset("./input/efsa-translated")















# ---  1. Continual pretraining of Llama-3-8b on the EFSA dataset to create Llama3-EFSA expert ---



# Configuration parameters
model_name = "llama-3-8b-Instruct-bnb-4bit"
max_seq_length = 2048
dtype = None
load_in_4bit = True

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/"+model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Apply LoRA configuration
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)








alpaca_prompt = """{}"""



# Define prompt template for EFSA pretraining
alpaca_prompt = """{}"""  # Simple format using just the translated text

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    translates = examples["translate"]
    texts = []
    for translate in translates:
        text = alpaca_prompt.format(translate) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

efsa_dataset = efsa_dataset.map(formatting_prompts_func, batched=True)

# Train on EFSA dataset
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=efsa_dataset["train"],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
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

print("Starting EFSA pretraining...")
model.train()
trainer_stats = trainer.train()

# Save pretrained model
trainer.save_model("outputs/Pretrained_Llama_on_FinEventData")
print("EFSA pretraining completed and model saved.")









# ---  2. Fine-tuning the pretrained model on financial sentiment analysis datasets (FPB, FiQA-SA, TFNS) ---


# Fine-tuning on financial sentiment datasets
dataset_name = "FPB"  # Can be "FPB", "FiQA-SA", or "TFNS"

# Label mappings for sentiment datasets
id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {"negative": 0, "neutral": 1, "positive": 2}

# Load appropriate sentiment dataset
if dataset_name == "FPB":
    sentiment_dataset = load_dataset('./input/finhmoe-datasets/financial_phrasebank-sentences_50agree_processed')
elif dataset_name == "FiQA-SA":
    sentiment_dataset = load_dataset('./input/finhmoe-datasets/fiqa-2018_processed')
elif dataset_name == "TFNS":
    sentiment_dataset = load_dataset('./input/finhmoe-datasets/twitter-financial-news-sentiment_processed')
else:
    raise ValueError("Invalid dataset name provided.")

print(f"Loaded {dataset_name} dataset successfully.")

# Define prompt template for sentiment fine-tuning
alpaca_prompt = """### Input:
{}

### Response:
{}"""

my_instruction = "What is the sentiment of this news/tweet? Please choose an answer from {negative/neutral/positive}."

def formatting_prompts_func(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for input_text, output in zip(inputs, outputs):
        text = alpaca_prompt.format(input_text, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

sentiment_dataset = sentiment_dataset.map(formatting_prompts_func, batched=True)

# Set training epochs based on dataset
if dataset_name == "FPB":
    num_train_epochs_trainer = 1
elif dataset_name == "FiQA-SA":
    num_train_epochs_trainer = 2
elif dataset_name == "TFNS":
    num_train_epochs_trainer = 1

# Fine-tune on sentiment dataset
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=sentiment_dataset["train"],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        #max_steps=10,
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

print(f"Starting fine-tuning on {dataset_name} dataset...")
model.train()
trainer_stats = trainer.train()
print(f"Fine-tuning on {dataset_name} completed.")


print("Expert training completed successfully.")







# --- Inference on Test Set ---
print("Running inference on test set...")
model = model.eval()
FastLanguageModel.for_inference(model)

# Define the prompt template for inference
def formatting_prompts_func_inference(examples):
    inputs = examples["input"]
    texts = []
    for input_text in inputs:
        # Include only the input part, leave the response part empty
        text = alpaca_prompt.format(input_text, "")  
        texts.append(text)
    return {"text": texts}

# Apply prompt template to the test set
sentiment_dataset_inference = sentiment_dataset.map(formatting_prompts_func_inference, batched=True)

# Tokenization function
def tokenize_function(data):
    return tokenizer(data["text"], truncation=True, padding=True, max_length=max_seq_length)

# Tokenize the dataset
tokenized_dataset = sentiment_dataset_inference.map(tokenize_function, batched=True)
print("Tokenization complete!")

# Custom dataset class
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_mask[idx]),
        }
    
    def __len__(self):
        return len(self.input_ids)

# Create test set data loader
test_dataset = MyDataset(tokenized_dataset["test"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=8, collate_fn=data_collator)

# --- Extract Hidden States (Multi-View Representations) ---
print("Extracting hidden states for multi-view representations...")
model = model.eval()
FastLanguageModel.for_inference(model)

# Generate save paths
model_name = "Pretrained_Llama_on_FinEventData"
extracted_model_name = model_name.split('/')[-1] if '/' in model_name else model_name
train_hidden_path = f"outputs_hiddens_{dataset_name}_{extracted_model_name}_train.pth"
test_hidden_path = f"outputs_hiddens_{dataset_name}_{extracted_model_name}_test.pth"



# Extract hidden states from training set
outputs_hiddens_train = []
with torch.no_grad():
    # Create training set data loader
    train_dataset = MyDataset(tokenized_dataset["train"])
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=8, collate_fn=data_collator)
    
    for batch in tqdm(train_dataloader, desc="Extracting train hidden states"):
        inputs = {k: v.to("cuda") for k, v in batch.items()}
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        # Extract the hidden state of the last token
        last_hidden_state = outputs.hidden_states[-1][:, -1, :].detach().cpu()
        outputs_hiddens_train.append(last_hidden_state)
    
outputs_hiddens_train = torch.cat(outputs_hiddens_train, dim=0)
torch.save(outputs_hiddens_train, train_hidden_path)
print(f"Training hidden states shape: {outputs_hiddens_train.shape}, saved to {train_hidden_path}")

# Extract hidden states from test set
outputs_hiddens_test = []
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Extracting test hidden states"):
        inputs = {k: v.to("cuda") for k, v in batch.items()}
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :].detach().cpu()
        outputs_hiddens_test.append(last_hidden_state)
    
outputs_hiddens_test = torch.cat(outputs_hiddens_test, dim=0)
torch.save(outputs_hiddens_test, test_hidden_path)
print(f"Test hidden states shape: {outputs_hiddens_test.shape}, saved to {test_hidden_path}")

print("Hidden state extraction completed successfully!")


    
    
