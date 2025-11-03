#!/usr/bin/env python
# coding: utf-8

"""
Hierarchical Expert Fusion for Financial Sentiment Analysis (HSMoE-FSA)

This module implements the Hierarchical Sparse Mixture-of-Experts framework
proposed in the paper 'Augmenting Large Language Models for Financial Sentiment Analysis'.
The architecture consists of three main components:
1. Expert Decision Network: Processes financial text through dimension-specialized experts
   (e.g., semantic, numerical, event, sentiment) to generate expert-specific predictions.
2. Heuristic Routing Network: Dynamically selects top-K most confident experts using
   entropy-based confidence scoring and assigns adaptive weights via sparse routing.
3. Mixture Inference Network: Fuses expert predictions using routing weights, enhances
   the fused representation, and produces the final calibrated sentiment distribution.

The model employs a two-phase optimization strategy:
- Phase 1: Specialize each expert independently with frozen routing/mixture components.
- Phase 2: Freeze experts and train routing & mixture networks for collaborative inference.

Key features:
- Entropy-based heuristic routing for interpretable expert selection
- Confidence-weighted fusion of multi-dimensional financial semantics
- Residual-enhanced prediction refinement
- Modular design supporting dynamic integration of domain-specialized experts
"""

import random
import torch
import numpy as np
import os
import datasets
from datasets import load_dataset, Dataset, DatasetDict
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import json
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from math import ceil
import sys


def seed_torch(seed=42):
    """
    Set random seeds for reproducibility across various libraries
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Set random seed for reproducible experiments
seed = 4
seed_torch(seed=seed)

# =============================================================================
# DATA CONFIGURATION AND LOADING
# =============================================================================
# Configuration for dataset selection
dataset_name = "FPB"  # Options: "FPB", "FiQA-SA", "TFNS"
top_k = 6  # Number of experts to activate in sparse routing
model_type_selected = "All"  # Expert selection mode

# Label mappings for sentiment classification
id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {"negative": 0, "neutral": 1, "positive": 2}

# Load financial sentiment dataset based on configuration
if dataset_name == "FPB":
    my_dataset = load_dataset('./input/finhmoe-datasets/financial_phrasebank-sentences_50agree_processed')
elif dataset_name == "FiQA-SA":
    my_dataset = load_dataset('./input/finhmoe-datasets/fiqa-2018_processed')
elif dataset_name == "TFNS":
    my_dataset = load_dataset('./input/finhmoe-datasets/twitter-financial-news-sentiment_processed')

# Process label strings to numerical format
labels_train = my_dataset["train"]["output"]
labels_test = my_dataset["test"]["output"]
labels_train = [label2id[label] for label in labels_train]
labels_test = [label2id[label] for label in labels_test]
labels_train = torch.tensor(labels_train, dtype=torch.long)
labels_test = torch.tensor(labels_test, dtype=torch.long)

# =============================================================================
# EXPERT MODEL CONFIGURATION
# =============================================================================
# Select which expert types to include based on ablation study
selected_types = ["Semantic", "Math", "Event", "Sentiment"]
if model_type_selected == "woSemantic":
    selected_types = ["Math", "Event", "Sentiment"]
elif model_type_selected == "woNumerical":
    selected_types = ["Semantic", "Event", "Sentiment"]
elif model_type_selected == "woEvent":
    selected_types = ["Semantic", "Math", "Sentiment"]
elif model_type_selected == "woSentiment":
    selected_types = ["Semantic", "Math", "Event"]

# Categorize expert models by their specialization domains
ModelCategory = {
    "Semantic": [
        "finbert_TRC2",
        "llama-3-8b-Instruct-bnb-4bit",
        "Qwen2.5-7B-Instruct-bnb-4bit",
        "gemma-2-9b-it-bnb-4bit",
        "mistral-7b-instruct-v0.3-bnb-4bit",
        "Phi-3.5-mini-instruct-bnb-4bit",
    ],
    "Math": [
        "Qwen2.5-Math-7B-Instruct-bnb-4bit",
        "Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "codellama-7b-bnb-4bit",
        "codegemma-7b-it-bnb-4bit",
        "Pretrained_Llama_on_FinanceMATH",
    ],
    "Event": [
        "llama-EventPrompt",
        "Pretrained_Llama_on_FinEventData",
    ],
    "Sentiment": [
        "finbert_FPB",
        "Pretrained_Llama_on_4SentiTrainData",
    ]
}

# Create list of model names based on selected expert types
model_name_list = []
for k, v in ModelCategory.items():
    if k in selected_types:
        model_name_list.extend(v)
model_name_list = [dataset_name + "_" + i for i in model_name_list]

# =============================================================================
# LOAD EXPERT HIDDEN STATES (PRE-COMPUTED REPRESENTATIONS)
# =============================================================================
# Load pre-computed hidden states from various expert models
outputs_hiddens_path = "./input/outputs-hiddens-" + dataset_name.lower() + "/outputs_hiddens/"
outputs_hiddens_train_list = []
outputs_hiddens_test_list = []
outputs_hiddens_dim_list = []
idx_to_expert_name = {}  # Mapping from index to expert model name

for idx, model_name in enumerate(model_name_list):
    if '/' in model_name:
        extracted_content = model_name.split('/')[-1]
    else:
        extracted_content = model_name
    idx_to_expert_name[idx] = extracted_content
    
    # Load train and test hidden states for each expert
    outputs_hiddens_train_pth = outputs_hiddens_path + "outputs_hiddens_" + extracted_content + "_train.pth"
    outputs_hiddens_test_pth = outputs_hiddens_path + "outputs_hiddens_" + extracted_content + "_test.pth"
    
    outputs_hiddens_train = torch.load(outputs_hiddens_train_pth, weights_only=True).squeeze().float()
    outputs_hiddens_test = torch.load(outputs_hiddens_test_pth, weights_only=True).squeeze().float()
    
    outputs_hiddens_train_list.append(outputs_hiddens_train)
    outputs_hiddens_test_list.append(outputs_hiddens_test)
    
    _, hidden_dim = outputs_hiddens_train.shape
    outputs_hiddens_dim_list.append(hidden_dim)

# Concatenate expert outputs along feature dimension
outputs_hiddens_train_cat = torch.cat(outputs_hiddens_train_list, dim=1)
outputs_hiddens_test_cat = torch.cat(outputs_hiddens_test_list, dim=1)

# Create PyTorch datasets and dataloaders
train_dataset = TensorDataset(outputs_hiddens_train_cat, labels_train)
test_dataset = TensorDataset(outputs_hiddens_test_cat, labels_test)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Save expert mapping for reference
with open("./working/idx_to_expert_name.json", 'w', encoding='utf-8') as f:
    json.dump(idx_to_expert_name, f, indent=4)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dims = outputs_hiddens_dim_list  # Input dimensions for each expert
num_heads = 8  # Number of attention heads
hidden_dim = 1024  # Hidden dimension size
num_experts = len(input_dims)  # Total number of experts
batch_size = 8  # Training batch size
labels_set = list(label2id.values())  # Available sentiment labels
num_classes = len(labels_set)  # Number of sentiment classes

# =============================================================================
# EXPERT DECISION NETWORK IMPLEMENTATION
# =============================================================================
class ExpertNetwork(nn.Module):
    """
    Expert Decision Network: Dimension-specialized sub-network with self-attention
    
    Each expert processes input through:
    1. Domain projection to shared latent space
    2. Multi-head self-attention for contextual interaction
    3. Feature fusion with residual connections
    4. Layer normalization for stability
    5. Expert-specific classification
    
    Implements the expert decision network described in section 4.3.1 of the paper.
    """
    def __init__(self, input_dim, hidden_dim, num_heads, num_classes):
        super(ExpertNetwork, self).__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        
        # Linear projection layer: Maps expert-specific features to shared space
        self.expert_hidden = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head self-attention: Captures contextual relationships
        self.self_attention = MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        
        # Feed-forward network: Enhances feature expressiveness
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),  # Gaussian Error Linear Units activation
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Layer normalization: Stabilizes training and preserves information
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Expert classifier: Produces dimension-specific sentiment predictions
        self.expert_classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Domain projection: Transform to shared semantic space
        x = self.expert_hidden(x)
        
        # Multi-head self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # Feature fusion with residual connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        
        # Classification: Generate expert-specific sentiment probabilities
        expert_class_logits = self.expert_classifier(x)
        expert_class_logits_log_softmax = F.log_softmax(expert_class_logits, dim=-1)
        
        # Entropy calculation for heuristic routing
        expert_entropy = F.softmax(expert_class_logits, dim=-1) * F.log_softmax(expert_class_logits, dim=-1)
        expert_entropy = -expert_entropy.sum(dim=-1)
        
        return x, expert_class_logits_log_softmax, expert_entropy


def freeze_net(net):
    """Freeze network parameters to prevent updates during training"""
    for param in net.parameters():
        param.requires_grad = False
        
def unfreeze_net(net):
    """Unfreeze network parameters to allow updates during training"""
    for param in net.parameters():
        param.requires_grad = True


def train_single_expert(expert, train_loader, optimizer, criterion, device, expert_index, max_steps=None):
    """
    Train a single expert network following Phase 1 optimization strategy
    
    Args:
        expert: The expert network to train
        train_loader: DataLoader with training data
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        device: Training device (CPU/GPU)
        expert_index: Index of the current expert
        max_steps: Maximum training steps (for partial training)
    
    Returns:
        Average training loss
    """
    expert.train()
    total_loss = 0
    step_counter = 0

    for inputs, labels in tqdm(train_loader, desc=f"Training expert {expert_index}"):
        if max_steps is not None and step_counter >= max_steps:
            break

        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Extract expert-specific inputs from concatenated features
        start_index = 0
        sub_vectors = []
        for length in input_dims:
            end_index = start_index + length
            sub_vector = inputs[:, start_index:end_index]
            sub_vectors.append(sub_vector)
            start_index = end_index
            
        inputs = sub_vectors[expert_index]

        # Forward pass through expert network
        expert_output, expert_class_logits_log_softmax, expert_entropy = expert(inputs)
        expert_loss = criterion(expert_class_logits_log_softmax, labels)

        # Backward pass and parameter update
        expert_loss.backward()
        optimizer.step()

        total_loss += expert_loss.item()
        step_counter += 1

    return total_loss / step_counter


def test_single_expert(expert, test_loader, criterion, device, expert_index):
    """
    Evaluate a single expert network on test data
    
    Args:
        expert: The expert network to evaluate
        test_loader: DataLoader with test data
        criterion: Loss function
        device: Evaluation device (CPU/GPU)
        expert_index: Index of the current expert
    
    Returns:
        test_loss: Average test loss
        test_accuracy: Classification accuracy
        test_single_y_true: Ground truth labels
        test_single_y_pred: Predicted labels
    """
    expert.eval()
    total_loss = 0
    total_correct = 0
    test_single_y_true = []
    test_single_y_pred = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Testing expert {expert_index}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            test_single_y_true.extend(labels.cpu().numpy())

            # Extract expert-specific inputs
            start_index = 0
            sub_vectors = []
            for length in input_dims:
                end_index = start_index + length
                sub_vector = inputs[:, start_index:end_index]
                sub_vectors.append(sub_vector)
                start_index = end_index
                
            inputs = sub_vectors[expert_index]

            # Forward pass
            expert_output, expert_class_logits_log_softmax, expert_entropy = expert(inputs)
            expert_loss = criterion(expert_class_logits_log_softmax, labels)

            # Calculate accuracy
            _, predicted = torch.max(expert_class_logits_log_softmax, -1)
            test_single_y_pred.extend(predicted.cpu().numpy())
            total_correct += (predicted == labels).sum().item()
            total_loss += expert_loss.item()

    return total_loss / len(test_loader), total_correct / len(test_loader.dataset), np.array(test_single_y_true), np.array(test_single_y_pred)


# =============================================================================
# PHASE 1: EXPERT NETWORK SPECIALIZATION
# =============================================================================
# Initialize expert networks
experts = nn.ModuleList([ExpertNetwork(input_dim, hidden_dim, num_heads, num_classes) for input_dim in input_dims])
experts.to(device)

# Expert training parameters (dataset-specific)
if dataset_name == "FPB":
    num_expert_epochs_initial = 2
    max_steps_per_expert_initial = None
    learning_rate_per_expert_initial = 0.00001
elif dataset_name == "FiQA-SA":
    num_expert_epochs_initial = 3
    max_steps_per_expert_initial = None
    learning_rate_per_expert_initial = 0.00001
elif dataset_name == "TFNS":
    num_expert_epochs_initial = 4
    max_steps_per_expert_initial = None
    learning_rate_per_expert_initial = 0.000001

experts_params = {
    i: {'num_expert_epochs': num_expert_epochs_initial, 
        'max_steps_per_expert': max_steps_per_expert_initial, 
        'learning_rate_per_expert': learning_rate_per_expert_initial}
    for i in range(num_experts)
}

# Train each expert independently (Phase 1 optimization)
criterion = nn.CrossEntropyLoss()
results_per_expert = pd.DataFrame(columns=["Expert_Index", "Expert_Name", "Accuracy", "F1 Macro", "F1 Micro", "F1 Weighted", "Classification Report", "Confusion Matrix"])
F1_Weighted_experts = {}

for expert_index, expert_name in idx_to_expert_name.items():
    expert_params = experts_params.get(expert_index, {'num_expert_epochs': 2, 'max_steps_per_expert': None, 'learning_rate_per_expert': 0.00001})
    
    num_expert_epochs = expert_params['num_expert_epochs']
    max_steps_per_expert = expert_params['max_steps_per_expert']
    learning_rate_per_expert = expert_params['learning_rate_per_expert']

    # Create optimizer for current expert
    optimizer = torch.optim.Adam(experts[expert_index].parameters(), lr=learning_rate_per_expert)
    
    # Freeze other experts to prevent interference (Phase 1 strategy)
    for i, expert in enumerate(experts):
        if i != expert_index:
            freeze_net(expert)

    # Training loop for current expert
    epoch_counter = 0
    step_counter = 0
    while (max_steps_per_expert is None or step_counter < max_steps_per_expert) and (num_expert_epochs is None or epoch_counter < num_expert_epochs):
        max_steps = (max_steps_per_expert - step_counter) if max_steps_per_expert is not None else None
        train_loss = train_single_expert(experts[expert_index], train_loader, optimizer, criterion, device, expert_index, max_steps=max_steps)
        test_loss, test_accuracy, test_single_y_true, test_single_y_pred = test_single_expert(experts[expert_index], test_loader, criterion, device, expert_index)
        print(f'Expert {expert_index} {expert_name} Epoch {epoch_counter+1}/{num_expert_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        
        epoch_counter += 1
        if max_steps_per_expert is not None:
            step_counter += len(train_loader) if step_counter + len(train_loader) <= max_steps_per_expert else max_steps_per_expert - step_counter
        else:
            step_counter += len(train_loader)
        if max_steps_per_expert is not None and step_counter >= max_steps_per_expert:
            break

    # Unfreeze all experts after training current one
    for expert in experts:
        unfreeze_net(expert)
    
    # Calculate evaluation metrics
    y_true, y_pred = test_single_y_true, test_single_y_pred
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    
    # Store results
    cr = classification_report(y_true, y_pred, digits=5, labels=labels_set)
    cm = confusion_matrix(y_true, y_pred, labels=labels_set)
    cr_output_dict = classification_report(y_true, y_pred, digits=5, labels=labels_set, output_dict=True)
    f1_scores_per_class = {f"F1 {label}": cr_output_dict[str(label)]['f1-score'] for label in labels_set}
    
    results_per_expert = pd.concat([results_per_expert, pd.DataFrame({
        "Expert_Index": [f"Expert {expert_index} "],
        "Expert_Name": [expert_name],
        "Accuracy": [acc],
        "F1 Macro": [f1_macro],
        "F1 Micro": [f1_micro],
        "F1 Weighted": [f1_weighted],
        **f1_scores_per_class,
        "Classification Report": [cr],
        "Confusion Matrix": [cm]
    })], ignore_index=True)

    F1_Weighted_experts[expert_name] = f1_weighted

# Save expert results
excel_file_path = './working/results_per_expert_'+ dataset_name + '.xlsx'
results_per_expert.to_excel(excel_file_path, index=False)

# =============================================================================
# HEURISTIC ROUTING NETWORK IMPLEMENTATION
# =============================================================================
class Router_Network(nn.Module):
    """
    Heuristic Routing Network: Entropy-based sparse routing mechanism
    
    Implements two routing strategies:
    1. Parameter-based routing (vanilla): Uses trainable parameters
    2. Entropy-based routing (heuristic): Uses prediction confidence scores
    
    Corresponds to section 4.3.2 in the paper.
    """
    def __init__(self, num_classes, num_experts, top_k):
        super(Router_Network, self).__init__()
        self.top_k = top_k  # Number of experts to activate
        self.w_g = nn.Parameter(torch.randn(num_classes, 1))  # Trainable parameter for vanilla routing
        self.entropy_database = torch.zeros(num_experts, 0)  # Storage for entropy values (unused in forward)
        self.num_experts = num_experts

    def forward(self, expert_class_logits, expert_entropies):
        # Parameter-based routing (vanilla MoE approach)
        logits = torch.matmul(expert_class_logits, self.w_g)
        logits = logits.squeeze(-1)
        top_k_logits, indices_vanillabased = logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices_vanillabased, top_k_logits)
        router_output_vanillabased = F.softmax(sparse_logits, dim=-1)
        
        # Entropy-based routing (heuristic approach)
        # Lower entropy indicates higher confidence -> higher weight
        weights = F.softmax(-expert_entropies, dim=1)
        top_k_weights, indices_entropybased = torch.topk(weights, self.top_k, dim=1)
        zeros = torch.full_like(weights, float('-inf'))
        sparse_weights = zeros.scatter(-1, indices_entropybased, top_k_weights)
        router_output_entropybased = F.softmax(sparse_weights, dim=-1)
        
        return router_output_vanillabased, router_output_entropybased, indices_vanillabased, indices_entropybased


# =============================================================================
# MIXTURE INFERENCE NETWORK IMPLEMENTATION
# =============================================================================
class MixerNetwork(nn.Module):
    """
    Mixture Inference Network: Dynamic integration of expert predictions
    
    Combines expert predictions using routing weights and applies:
    1. Weighted combination of expert predictions
    2. Semantic space projection
    3. Residual enhancement
    4. Final classification
    
    Corresponds to section 4.3.3 in the paper.
    """
    def __init__(self, num_classes, num_experts, top_k, router_model):
        super(MixerNetwork, self).__init__()
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_model = router_model  # Routing strategy: "vanilla", "entropy", or "vanilla_plus_entropy"
        self.router = Router_Network(self.num_classes, self.num_experts, self.top_k)
        self.classifier_mixer = nn.Linear(self.num_classes, self.num_classes)  # Final classification layer
        self.layer_norm = nn.LayerNorm(self.num_classes)  # Layer normalization for stability

    def forward(self, x):
        # Split concatenated input into expert-specific components
        start_index = 0
        sub_vectors = []
        for length in input_dims:
            end_index = start_index + length
            sub_vector = x[:, start_index:end_index]
            sub_vectors.append(sub_vector)
            start_index = end_index
        
        # Process each input through its corresponding expert
        expert_outputs = []
        expert_class_logits = []
        expert_entropies = []
        for expert, xi in zip(experts, sub_vectors):
            expert_output, expert_class_logit, expert_entropy = expert(xi)
            expert_outputs.append(expert_output)
            expert_class_logits.append(expert_class_logit)
            expert_entropies.append(expert_entropy)
        
        # Stack expert outputs for batch processing
        expert_outputs = torch.stack(expert_outputs)
        expert_class_logits = torch.stack(expert_class_logits)
        expert_entropies = torch.stack(expert_entropies)
        
        expert_outputs = expert_outputs.permute(1, 0, 2)
        expert_class_logits = expert_class_logits.permute(1, 0, 2)
        expert_entropies = expert_entropies.permute(1, 0)

        # Routing: Select top-K experts and assign weights
        router_output_vanillabased, router_output_entropybased, indices_vanillabased, indices_entropybased = self.router(expert_class_logits, expert_entropies)

        # Combine routing strategies based on configuration
        if self.router_model == "vanilla":
            router_output = router_output_vanillabased
        elif self.router_model == "entropy":
            router_output = router_output_entropybased
        elif self.router_model == "vanilla_plus_entropy":
            router_output = router_output_vanillabased + router_output_entropybased

        # Weighted combination of expert predictions
        weighted_probabilities = torch.matmul(router_output.unsqueeze(1), expert_class_logits).squeeze(1)

        # Final classification with residual enhancement
        z = self.classifier_mixer(weighted_probabilities)
        z = self.layer_norm(weighted_probabilities + z)  # Residual connection
        mixed_log_probabilities = F.log_softmax(z, dim=-1)

        return mixed_log_probabilities, expert_class_logits, (router_output_vanillabased, router_output_entropybased, indices_vanillabased, indices_entropybased)


def train_mixer(mixer, train_loader, optimizer, criterion, device):
    """
    Train the mixture network (Phase 2 optimization)
    
    Args:
        mixer: Mixture network to train
        train_loader: Training data loader
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        device: Training device (CPU/GPU)
    
    Returns:
        Average training loss
    """
    mixer.train()
    total_loss = 0
    for inputs, labels in tqdm(train_loader, desc=f"Training mixer"):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass through mixture network
        outputs, expert_class_logits, _ = mixer(inputs)
        loss = criterion(outputs, labels)
        
        # Add expert losses for joint optimization
        expert_losses = [criterion(expert_class_logit, labels) for expert_class_logit in expert_class_logits.unbind(dim=1)]
        loss = loss + sum(expert_losses)

        # Backward pass and parameter update
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
            
    return total_loss / len(train_loader)


def test_mixer(mixer, test_loader, criterion, device):
    """
    Evaluate the mixture network on test data
    
    Args:
        mixer: Mixture network to evaluate
        test_loader: Test data loader
        criterion: Loss function
        device: Evaluation device (CPU/GPU)
    
    Returns:
        test_loss: Average test loss
        test_accuracy: Classification accuracy
        y_true: Ground truth labels
        y_pred: Predicted labels
    """
    mixer.eval()
    total_loss = 0
    total_correct = 0
    labels_list, prediction_list = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing mixer"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs, expert_class_logits, _ = mixer(inputs)
            loss = criterion(outputs, labels)
            expert_losses = [criterion(expert_class_logit, labels) for expert_class_logit in expert_class_logits.unbind(dim=1)]
            loss = loss + sum(expert_losses)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            labels_list.extend(labels.tolist())
            prediction_list.extend(predicted.tolist())
            total_loss += loss.item()
            
    y_true = labels_list
    y_pred = prediction_list
    return total_loss / len(test_loader), total_correct / len(test_loader.dataset), y_true, y_pred

# =============================================================================
# PHASE 2: ROUTING-MIXTURE COORDINATION
# =============================================================================
# Train mixture networks with different routing strategies
router_models = ["entropy"]  # Options: ["vanilla", "entropy", "vanilla_plus_entropy"]

# Dataset-specific training parameters
if dataset_name == "FPB":
    num_mixer_epochs = 6
    learning_rate_mixer = 0.001
elif dataset_name == "FiQA-SA":
    num_mixer_epochs = 6
    learning_rate_mixer = 0.001
elif dataset_name == "TFNS":
    num_mixer_epochs = 6
    learning_rate_mixer = 0.0005

for router_model in router_models:
    # Initialize mixture network with selected routing strategy
    mixer = MixerNetwork(num_classes, num_experts, top_k, router_model)
    mixer.to(device)
    
    # Freeze experts and unfreeze mixer (Phase 2 strategy)
    freeze_net(experts)
    unfreeze_net(mixer)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mixer.parameters(), lr=learning_rate_mixer)
    
    # Training loop for mixture network
    results_mixer = pd.DataFrame(columns=["Epoch", "Accuracy", "F1 Macro", "F1 Micro", "F1 Weighted", "Classification Report", "Confusion Matrix"])
    
    for epoch in range(num_mixer_epochs):
        train_loss = train_mixer(mixer, train_loader, optimizer, criterion, device)
        test_loss, test_accuracy, y_true, y_pred = test_mixer(mixer, test_loader, criterion, device)
        print(f'Epoch {epoch+1}/{num_mixer_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

        # Calculate evaluation metrics
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_micro = f1_score(y_true, y_pred, average="micro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        
        # Store results
        cr = classification_report(y_true, y_pred, digits=5, labels=labels_set)
        cm = confusion_matrix(y_true, y_pred, labels=labels_set)
        cr_output_dict = classification_report(y_true, y_pred, digits=5, labels=labels_set, output_dict=True)
        f1_scores_per_class = {f"F1 {label}": cr_output_dict[str(label)]['f1-score'] for label in labels_set}

        results_mixer = pd.concat([results_mixer, pd.DataFrame({
            "Epoch": [f"Epoch {epoch}"],
            "Accuracy": [acc],
            "F1 Macro": [f1_macro],
            "F1 Micro": [f1_micro],
            "F1 Weighted": [f1_weighted],
            **f1_scores_per_class,
            "Classification Report": [cr],
            "Confusion Matrix": [cm]
        })], ignore_index=True)

    # Save results
    excel_file_path = './working/results_mixer_'+ router_model + '_' + dataset_name + '.xlsx'
    results_mixer.to_excel(excel_file_path, index=False)
    
    # Save detailed classification report
    output_path = './working/results_ClassificationReport_'  + dataset_name + "_HMoE_" + router_model + '.txt'
    with open(output_path, 'w') as f:
        f.write(f"Acc: {acc:.5f}. F1 macro: {f1_macro:.5f}. F1 micro: {f1_micro:.5f}. F1 weighted: {f1_weighted:.5f}.\n")
        f.write(f"Classification_Report:\n {cr}.\n")
        f.write(f"Confusion_Matrix:\n {cm}.")

# =============================================================================
# SAVE MODEL PARAMETERS AND CONFIGURATION
# =============================================================================
# Save all model parameters and configuration for reproducibility
parameters = {
    "dataset_name": dataset_name,
    "selected_types": selected_types,
    "id2label": id2label,
    "label2id": label2id,
    "input_dims": input_dims,
    "num_heads": num_heads,
    "hidden_dim": hidden_dim,
    "num_experts": num_experts,
    "batch_size": batch_size,
    "labels_set": labels_set,
    "num_classes": num_classes,
    "idx_to_expert_name": idx_to_expert_name,
    "num_expert_epochs_initial": num_expert_epochs_initial,
    "max_steps_per_expert_initial": max_steps_per_expert_initial,
    "learning_rate_per_expert_initial": learning_rate_per_expert_initial,
    "router_model": router_model,
    "top_k": top_k,
    "seed": seed,
    "num_mixer_epochs": num_mixer_epochs,
    "learning_rate_mixer": learning_rate_mixer
}

json_file_path = './working/model_parameters.json'
with open(json_file_path, 'w') as json_file:
    json.dump(parameters, json_file, indent=4)

print("Training completed successfully!")