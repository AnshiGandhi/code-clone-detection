import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.model_selection import train_test_split
model_path = "final/fine_tuned_model_final"

# Load the fine-tuned model and tokenizer
loaded_model = BertForSequenceClassification.from_pretrained(model_path)
loaded_tokenizer = BertTokenizer.from_pretrained(model_path)

print("Model and tokenizer loaded successfully!")

# Generate embeddings for candidate snippets in batches
batch_size = 32
candidate_embeddings = []

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset

# Specify the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assume get_batch_embeddings moves data to the device
def get_batch_embeddings(batch):
    # Tokenize and move to device
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        # Get outputs from the base BERT model
        outputs = model.bert(**inputs)  # Access the underlying BERT model
    # Return CLS token embeddings
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


# Compute dataset embeddings
def compute_dataset_embeddings(dataset):
    snippets = [example['code'] for example in dataset]
    embeddings = []
    print("In compute_dataset_embeddings")
    for i in range(0, len(snippets), batch_size):
        batch = snippets[i:i + batch_size]
        batch_embeddings = get_batch_embeddings(batch)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings), [example['label'] for example in dataset]

# MAP@R calculation remains unchanged
def calculate_map_at_r(query_embeddings, query_labels, candidate_embeddings, candidate_labels):
    map_scores = []
    print("Calculate MAP@R")
    for i, query_embedding in enumerate(query_embeddings):
        query_label = query_labels[i]

        # Compute cosine similarities with candidates
        similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]

        # Sort candidates by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_labels = [candidate_labels[j] for j in sorted_indices]

        # Identify relevant candidates
        relevant_indices = [j for j, label in enumerate(sorted_labels) if label == query_label]
        R = len(relevant_indices)

        # Skip if no relevant items
        if R == 0:
            map_scores.append(0.0)
            continue

        # Calculate precision at K
        precisions = []
        for k, index in enumerate(relevant_indices):
            relevant_count = k + 1
            precision_at_k = relevant_count / (index + 1)
            precisions.append(precision_at_k)

        # Average Precision for this query
        average_precision = sum(precisions) / R
        map_scores.append(average_precision)

    # Mean Average Precision across all queries
    return np.mean(map_scores) if map_scores else 0.0

# Load the dataset from the JSON file
train_dataset = load_dataset('json', data_files='final/train_dataset.json')['train']
test_dataset = load_dataset('json', data_files='final/test_dataset.json')['train']
valid_dataset = load_dataset('json', data_files='final/valid_dataset.json')['train']

# Step 3: Shuffle and select 100 random samples
# train_dataset = train_dataset.shuffle(seed=42).select(range(4000))
# valid_dataset = valid_dataset.shuffle(seed=42).select(range(500))
# test_dataset = test_dataset.shuffle(seed=42).select(range(500))

# Compute embeddings
train_embeddings, train_labels = compute_dataset_embeddings(train_dataset)
test_embeddings, test_labels = compute_dataset_embeddings(test_dataset)
valid_embeddings, valid_labels = compute_dataset_embeddings(valid_dataset)

# Updated MAP@R evaluation
train_map_at_r = calculate_map_at_r(train_embeddings, train_labels, train_embeddings, train_labels)
test_map_at_r = calculate_map_at_r(test_embeddings, test_labels, train_embeddings, train_labels)
valid_map_at_r = calculate_map_at_r(valid_embeddings, valid_labels, train_embeddings, train_labels)

print(f"MAP@R for Train Data: {train_map_at_r:.4f}")
print(f"MAP@R for Test Data: {test_map_at_r:.4f}")
print(f"MAP@R for Valid Data: {valid_map_at_r:.4f}")