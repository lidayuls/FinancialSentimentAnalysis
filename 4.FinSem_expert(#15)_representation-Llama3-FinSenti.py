"""

Description:
This script implements the **Financial Implicit Sentiment Decoding Expert (FinSent) (Llama3-FinSenti) ** 
module of the HSMoE-FSA framework, as described in the paper 
"Augmenting Large Language Models for Financial Sentiment Analysis: A Heuristic Sparse Mixture-of-Experts Framework".



This module constructs the Llama3-FinSenti expert, a specialized component within the HSMoE-FSA framework designed to decode implicit sentiment expressions in financial texts. 
It follows a two-stage domain-adaptive training strategy:
1. Continual pretraining on a merged corpus of multiple financial sentiment datasets to strengthen contextual understanding.
2. Instruction fine-tuning on specific financial sentiment analysis benchmarks (FPB, FiQA-SA, TFNS) to enhance classification accuracy.
Finally, hidden states are extracted from both training and test sets to generate multi-view representations for integration into the HMoE).


"""







# !pip show unsloth

import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    import torch; v = re.match(r"[0-9\.]{3,}", str(torch.__version__)).group(0)
    xformers = "xformers==" + ("0.0.32.post2" if v == "2.8.0" else "0.0.29.post3")
    !pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth
!pip install transformers==4.55.4


#%%capture
!pip uninstall unsloth -y
!pip install --no-cache-dir "unsloth @ git+https://github.com/lidayuls/unsloth.git"














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










#  STEP 1: CONTINUAL PRETRAINING ON MULTIPLE FINANCIAL SENTIMENT DATASETS

# ---  1. Continual pretraining of Llama-3-8b on the  multiple financial sentiment analysis dataset to further strengthen contextual understanding of financial sentiments---


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











# Load four financial sentiment datasets from local paths 
# Each dataset has been preprocessed and saved in a structured format
my_dataset_FPB = load_dataset('./input/finhmoe-datasets/financial_phrasebank-sentences_50agree_processed')
my_dataset_FiQA = load_dataset('./input/finhmoe-datasets/fiqa-2018_processed')
my_dataset_TFNS = load_dataset('./input/finhmoe-datasets/news_with_gpt_instructions_processed')
my_dataset_NWGI = load_dataset('./input/finhmoe-datasets/twitter-financial-news-sentiment_processed')

# Combine all training sets into a single training dataset
train_datasets = [
    my_dataset_FPB['train'],
    my_dataset_FiQA['train'],
    my_dataset_TFNS['train'],
    my_dataset_NWGI['train']
]
merged_train_dataset = datasets.concatenate_datasets(train_datasets)

# Combine all test sets into a single test dataset
test_datasets = [
    my_dataset_FPB['test'],
    my_dataset_FiQA['test'],
    my_dataset_TFNS['test'],
    my_dataset_NWGI['test']
]
merged_test_dataset = datasets.concatenate_datasets(test_datasets)

# Create a DatasetDict containing the merged train and test splits
my_dataset = datasets.DatasetDict({
    'train': merged_train_dataset,
    'test': merged_test_dataset
})


# Define a prompt template for instruction-tuning format

my_instruction = "What is the sentiment of this news/tweet? Please choose an answer from {negative/neutral/positive}."
alpaca_prompt = """### Instruction:
{}

### Input:
{}

### Response:
{}"""


EOS_TOKEN = tokenizer.eos_token




def formatting_prompts_func(examples):
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = alpaca_prompt.format(my_instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }




my_dataset = my_dataset.map(formatting_prompts_func, batched=True)





# Train on dataset
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=my_dataset["train"],
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

# Start continual pretraining

model.train()
trainer_stats = trainer.train()







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
model_name = "Pretrained_Llama_on_4SentiTrainData"
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


    
    