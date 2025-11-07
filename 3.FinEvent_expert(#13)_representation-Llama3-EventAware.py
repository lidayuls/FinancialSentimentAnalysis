"""

Description:
This script implements the **Financial Event Impact Assessment (FinEvent) (Llama3-EventAware) ** 
module of the HSMoE-FSA framework, as described in the paper 
"Augmenting Large Language Models for Financial Sentiment Analysis: A Heuristic Sparse Mixture-of-Experts Framework".


Instruction-tuned with event-aware prompts

The code performs the following steps:
1. Prepares financial sentiment datasets (FPB, FiQA-SA, TFNS) with an event-aware instruction prompt.
2. Loads a pre-trained Llama-3 model using Unsloth for efficient fine-tuning.
3. Applies LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
4. Fine-tunes the model using SFTTrainer from TRL.
5. Performs inference on the test set and extracts predictions.
7. Extracts final hidden states for multi-view representation (used in later fusion stages).


"""








import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

!pip install -U uv --upgrade-strategy eager

!uv pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121

!uv pip install --no-deps \
  transformers==4.47.1 tokenizers==0.21.0 \
  huggingface-hub==0.26.2 \
  accelerate==1.0.0 \
  datasets==2.21.0 safetensors==0.4.4 sentencepiece==0.2.0 "numpy<2"

!uv pip install --no-cache-dir --no-deps "git+https://github.com/lidayuls/unsloth.git@mian-2025.1.14"
!uv pip install --no-cache-dir --no-deps unsloth-zoo==2025.1.5

!uv pip install bitsandbytes==0.45.1 --no-deps -v
!uv pip install trl==0.11.4 --no-deps -v

!uv pip install --no-deps xformers==0.0.29.post1

    

from unsloth import FastLanguageModel
import os
import random
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
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
from unsloth import is_bfloat16_supported


# --- Configuration ---
# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


    





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






# Fine-tuning on financial sentiment datasets
dataset_name = "FPB"  # Can be "FPB", "FiQA-SA", or "TFNS"

# Label mappings for sentiment datasets
id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {"negative": 0, "neutral": 1, "positive": 2}

# Load appropriate sentiment dataset
if dataset_name == "FPB":
    sentiment_dataset = load_dataset('/kaggle/input/finhmoe-datasets/financial_phrasebank-sentences_50agree_processed')
elif dataset_name == "FiQA-SA":
    sentiment_dataset = load_dataset('/kaggle/input/finhmoe-datasets/fiqa-2018_processed')
elif dataset_name == "TFNS":
    sentiment_dataset = load_dataset('/kaggle/input/finhmoe-datasets/twitter-financial-news-sentiment_processed')
else:
    raise ValueError("Invalid dataset name provided.")

print(f"Loaded {dataset_name} dataset successfully.")









# Configure event-aware instruction prompt
event_instruction = '''
Market sentiment is often influenced by corporate events, with beneficial occurrences prompting positive market sentiment and detrimental events precipitating negative sentiment. 
Please analyze the following financial text and evaluate the potential impact of the financial events mentioned in the text on market sentiment.
You are required to select an appropriate response from the predefined set of options: {negative/neutral/positive}.
'''

# Template for formatting training examples
alpaca_prompt = """### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token


def formatting_prompts_func(examples):
    """Format dataset examples with event-aware instruction."""
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for input_text, output_text in zip(inputs, outputs):
        text = alpaca_prompt.format(event_instruction, input_text, output_text) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


# Apply formatting to dataset
sentiment_dataset = sentiment_dataset.map(formatting_prompts_func, batched=True)
print("Formatted dataset:", sentiment_dataset)
print("Sample formatted text:", sentiment_dataset["train"][0])





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
        #max_steps=5,
        num_train_epochs=num_train_epochs_trainer,
        learning_rate=2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
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
        text = alpaca_prompt.format(event_instruction, input_text, "")
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
model_name = "llama-EventPrompt"
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


    
    
    
    
    
