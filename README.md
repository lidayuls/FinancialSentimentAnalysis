# Augmenting Large Language Models for Financial Sentiment Analysis: A Heuristic Sparse Mixture-of-Experts Framework

## Description
This repository contains the official implementation for the paper **"Augmenting Large Language Models for Financial Sentiment Analysis: A Heuristic Sparse Mixture-of-Experts Framework"**. The project introduces **HSMoE-FSA**, a novel framework that enhances financial sentiment analysis by integrating four specialized experts: 
- **FinSem** for financial terminology comprehension  
- **FinNum** for quantitative reasoning  
- **FinEvent** for event impact assessment  
- **FinSent** for implicit sentiment detection

The framework uses an entropy-based routing mechanism to dynamically combine expert outputs, achieving state-of-the-art performance on multiple financial sentiment benchmarks.

## Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Datasets 2.10+
- Unsloth (for QLoRA fine-tuning)
- TRL (for SFT training)
- Scikit-learn
- NumPy
- tqdm

## Dataset Information
The framework is evaluated on three publicly available financial sentiment analysis datasets:

1. **Financial PhraseBank (FPB)**: 5,000 sentences from financial news with sentiment labels (positive/neutral/negative).
2. **FiQA-SA**: 1,213 financial news headlines and social media posts with continuous sentiment scores discretized into three classes.
3. **Twitter Financial News Sentiment (TFNS)**: 11,931 annotated financial tweets categorized as bullish, bearish, or neutral.

All datasets are available on Hugging Face Datasets and are automatically downloaded and preprocessed by the provided scripts.

## Code Information
The repository is organized into the following Python scripts:

### 0. Data Preparation
- **`financial_sentiment_dataset_preparation.py`**: Downloads, preprocesses, and standardizes all financial sentiment datasets.

### 1. FinSem Experts
- **`FinSem_expert(#1)_representation-FinBERT-TRC2.py`**: Implements FinBERT-TRC2 expert (financial semantic comprehension).
- **`FinSem_expert(#2~6)_representation-LlamaQwenGemmaPhiMistral.py`**: Implements Llama3, Qwen2.5, Gemma2, Phi3.5, and Mistral as FinSem experts.

### 2. FinNum Experts
- **`FinNum_expert(#7)_representation-Llama3-FinanceMath.py`**: Implements Llama3-FinanceMath expert (quantitative reasoning).
- **`FinNum_expert(#8~11)_representation-QwenmathQwencoderCodellamaCodegemma.py`**: Implements Qwen2.5-Math, Qwen2.5-Coder, CodeLlama, and CodeGemma as FinNum experts.

### 3. FinEvent Experts
- **`FinEvent_expert(#12)_representation-Llama3-EFSA.py`**: Implements Llama3-EFSA expert (event impact assessment via quintuple annotation).
- **`FinEvent_expert(#13)_representation-Llama3-EventAware.py`**: Implements Llama3-EventAware expert (event-aware instruction tuning).

### 4. FinSent Experts
- **`FinSent_expert(#14)_representation-FinBERT-FPB.py`**: Implements FinBERT-FPB expert (implicit sentiment decoding).
- **`FinSent_expert(#15)_representation-Llama3-FinSenti.py`**: Implements Llama3-FinSenti expert (multi-dataset instruction tuning).

### 5. Main Framework
- **`HeuristicSparseMixture-of-Experts.py`**: Implements the full HSMoE-FSA framework with:
  - Expert Decision Network
  - Entropy-based Heuristic Routing
  - Mixture Inference Network
  - Phased Optimization Strategy

## Usage Instructions

### Prerequisites
Organize datasets into `./data/` with subdirectories for each dataset (FPB, FiQA-SA, TFNS). The preprocessing script will handle this automatically.

### Installation
Install required dependencies:
```bash
pip install torch transformers datasets unsloth trl scikit-learn numpy tqdm
```

### Running the Code

#### 1. Preprocess Datasets
```bash
python financial_sentiment_dataset_preparation.py
```

#### 2. Train Individual Experts
Example for training FinSem Expert #1 (FinBERT-TRC2):
```bash
python FinSem_expert\(#1\)_representation-FinBERT-TRC2.py
```

Repeat for all expert scripts (adjust file names as needed).

#### 3. Run Full HSMoE-FSA Framework
```bash
python HeuristicSparseMixture-of-Experts.py
```

### Configuration

#### Hardware Requirements
- **GPUs**: 4×NVIDIA A100 80GB GPUs (recommended for full expert training)
- **VRAM**: Minimum 40GB per GPU for 7B models; 80GB for larger experts

#### Optimization Strategy
- **Quantization**: 4-bit NormalFloat (NF4) via QLoRA
- **Adapter**: Low-Rank Adaptation (LoRA)
- **Optimizer**: AdamW
- **Phased Training**: Expert specialization → Routing calibration

## Methodology
The HSMoE-FSA framework integrates:

1. **Dimension-Specialized Experts**: Four expert categories targeting financial semantics, numeric reasoning, event impacts, and implicit sentiment.
2. **Entropy-Based Routing**: Dynamically selects top-K experts using entropy-based confidence scores.
3. **Hierarchical Fusion**: Combines expert predictions through weighted aggregation and residual enhancement.
4. **Phased Optimization**: Decouples expert training from router calibration to ensure stability.



---




