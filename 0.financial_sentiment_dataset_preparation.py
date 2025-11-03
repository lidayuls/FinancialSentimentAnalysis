"""
Data Preprocessing Pipeline for Financial Sentiment Analysis Datasets

This script handles the downloading, processing, and standardization of multiple financial sentiment analysis datasets.
It performs the following steps:
1. Downloads specified datasets from Hugging Face Hub (if not already cached).
2. Processes each dataset to unify input/output format with consistent label schema:
   - Input: Raw financial text (news, tweets, etc.)
   - Output: Sentiment label mapped to {negative, neutral, positive}
3. Splits datasets into train/test sets (where not already provided).
4. Saves processed datasets in both Hugging Face Dataset format and JSON lines format for flexibility.

The processed datasets are used as input for training and evaluation in the MoE-based sentiment analysis framework.

Primary Datasets Processed:
- financial_phrasebank (FPB)
- fiqa-2018 (FiQA SA)
- twitter-financial-news-sentiment (TFNS)
- news_with_gpt_instructions (NWGI)
- TweetFinSent (manually downloaded and processed)

"""

import json
import re
import os
import random
import datasets
import pandas as pd
from glob import glob
from tqdm.notebook import tqdm
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoConfig


# ========================================
# 1. Define and Download Datasets
# ========================================

# List of datasets to download: (source, subset), local_save_name
DATASETS = [
    (('pauri32/fiqa-2018', None), 'fiqa-2018'),
    (('zeroshot/twitter-financial-news-sentiment', None), 'twitter-financial-news-sentiment'),
    (('oliverwang15/news_with_gpt_instructions', None), 'news_with_gpt_instructions'),
    (('takala/financial_phrasebank', 'sentences_50agree'), 'financial_phrasebank-sentences_50agree'),
]

def download(no_cache=False, data_dir=None):
    """
    Download and cache datasets from Hugging Face Hub.
    
    Args:
        no_cache (bool): If True, force re-download even if dataset exists.
        data_dir (str or Path): Directory to save datasets. Defaults to current directory.
    """
    if data_dir is None:
        data_dir = os.getcwd()
    else:
        data_dir = str(data_dir)

    for src, dest in DATASETS:
        save_path = os.path.join(data_dir, dest)
        if os.path.exists(save_path) and not no_cache:
            print(f"Dataset found at {save_path}, skipping download.")
            continue
        print(f"Downloading {src[0]}...")
        dataset = load_dataset(*src, trust_remote_code=True)
        dataset.save_to_disk(save_path)

# Execute download
download(no_cache=False)


# ========================================
# 2. Process Individual Datasets
# ========================================

# ------------------------
# Financial PhraseBank (FPB)
# ------------------------
print("\nProcessing Financial PhraseBank...")
label_map_fpb = {0: "negative", 1: "neutral", 2: "positive"}

# Load dataset from local disk
fpb_dataset = load_from_disk('./financial_phrasebank-sentences_50agree/')
fpb_dataset = fpb_dataset["train"].to_pandas()
fpb_dataset.columns = ["input", "output"]
fpb_dataset["output"] = fpb_dataset["output"].map(label_map_fpb)

# Convert back to Dataset and split
fpb_dataset = Dataset.from_pandas(fpb_dataset)
fpb_dataset = fpb_dataset.train_test_split(test_size=0.2, seed=42)

# Save processed dataset
save_dir = './financial_phrasebank-sentences_50agree_processed/'
os.makedirs(save_dir, exist_ok=True)
fpb_dataset['train'].to_pandas().to_json(os.path.join(save_dir, 'train.json'), orient='records', lines=True)
fpb_dataset['test'].to_pandas().to_json(os.path.join(save_dir, 'test.json'), orient='records', lines=True)
print(f"FPB processed: {fpb_dataset}")


# ------------------------
# FiQA Sentiment Analysis (FiQA SA)
# ------------------------
print("\nProcessing FiQA-2018...")
def score_to_label(score):
    """Convert sentiment score to categorical label."""
    if score < -0.1:
        return "negative"
    elif -0.1 <= score < 0.1:
        return "neutral"
    else:
        return "positive"

fiqa_dataset = load_from_disk('./fiqa-2018/')
fiqa_dataset = datasets.concatenate_datasets([
    fiqa_dataset["train"], 
    fiqa_dataset["validation"], 
    fiqa_dataset["test"]
]).to_pandas()

fiqa_dataset["output"] = fiqa_dataset["sentiment_score"].apply(score_to_label)
fiqa_dataset = fiqa_dataset[['sentence', 'output']]
fiqa_dataset.columns = ["input", "output"]

fiqa_dataset = Dataset.from_pandas(fiqa_dataset)
fiqa_dataset = fiqa_dataset.train_test_split(test_size=0.226, seed=42)

# Save processed dataset
save_dir = './fiqa-2018_processed/'
os.makedirs(save_dir, exist_ok=True)
fiqa_dataset['train'].to_pandas().to_json(os.path.join(save_dir, 'train.json'), orient='records', lines=True)
fiqa_dataset['test'].to_pandas().to_json(os.path.join(save_dir, 'test.json'), orient='records', lines=True)
print(f"FiQA SA processed: {fiqa_dataset}")


# ------------------------
# Twitter Financial News Sentiment (TFNS)
# ------------------------
print("\nProcessing Twitter-Financial-News-Sentiment...")
label_map_tfns = {0: "negative", 1: "positive", 2: "neutral"}

tfns_dataset = load_from_disk('./twitter-financial-news-sentiment')
train_df = tfns_dataset['train'].to_pandas()
test_df = tfns_dataset['validation'].to_pandas()

train_df['label'] = train_df['label'].map(label_map_tfns)
test_df['label'] = test_df['label'].map(label_map_tfns)

train_df.columns = ['input', 'output']
test_df.columns = ['input', 'output']

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tfns_dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

# Save processed dataset
save_dir = './twitter-financial-news-sentiment_processed/'
os.makedirs(save_dir, exist_ok=True)
tfns_dataset['train'].to_pandas().to_json(os.path.join(save_dir, 'train.json'), orient='records', lines=True)
tfns_dataset['test'].to_pandas().to_json(os.path.join(save_dir, 'test.json'), orient='records', lines=True)
print(f"TFNS processed: {tfns_dataset}")


# ------------------------
# News With GPT Instructions (NWGI)
# ------------------------
print("\nProcessing News With GPT Instructions...")
nwgi_dataset = load_from_disk('./news_with_gpt_instructions/')
train_df = nwgi_dataset['train'].to_pandas()
test_df = nwgi_dataset['test'].to_pandas()

train_df['output'] = train_df['label']
train_df['input'] = train_df['news']
train_df = train_df[['input', 'output']]

test_df['output'] = test_df['label']
test_df['input'] = test_df['news']
test_df = test_df[['input', 'output']]

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

nwgi_dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

# Save processed dataset
save_dir = './news_with_gpt_instructions_processed/'
os.makedirs(save_dir, exist_ok=True)
nwgi_dataset['train'].to_pandas().to_json(os.path.join(save_dir, 'train.json'), orient='records', lines=True)
nwgi_dataset['test'].to_pandas().to_json(os.path.join(save_dir, 'test.json'), orient='records', lines=True)
print(f"NWGI processed: {nwgi_dataset}")







