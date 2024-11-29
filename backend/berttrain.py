# -*- coding: utf-8 -*-
"""BertTrain.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sTG-pOoBcbOMaPgMaBil8yTuW2r62Cke
"""

pip install datasets

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/gdrive')

# Step 1: Load datasets for fine-tuning
train_dataset = load_dataset('json', data_files='/content/gdrive/MyDrive/final/train.jsonl')['train']

# Convert dataset to a list of dictionaries
train_dataset = train_dataset.to_list()

train_dataset, temp_data = train_test_split(train_dataset, test_size=0.2, random_state=42)
valid_dataset, test_dataset = train_test_split(temp_data, test_size=0.5, random_state=42)

# Convert list of dictionaries to DataFrame
train_df = pd.DataFrame(train_dataset)
valid_df = pd.DataFrame(valid_dataset)
test_df = pd.DataFrame(test_dataset)

# Convert DataFrame to Dataset object
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)
test_dataset = Dataset.from_pandas(test_df)

def process_labels(example):
    example['label'] = int(example['label'])  # Convert label from string to integer
    return example

train_dataset = train_dataset.map(process_labels)
valid_dataset = valid_dataset.map(process_labels)
test_dataset = test_dataset.map(process_labels)

# Save to Google Drive as JSON
train_dataset.to_json('/content/gdrive/MyDrive/final/train_dataset.json')
valid_dataset.to_json('/content/gdrive/MyDrive/final/valid_dataset.json')
test_dataset.to_json('/content/gdrive/MyDrive/final/test_dataset.json')

# Load BERT tokenizer and pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=105)

# Freeze lower BERT layers to speed up training
for param in model.bert.encoder.layer[:4].parameters():
    param.requires_grad = False


# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenize datasets
def tokenize_code(example):
    return tokenizer(
        example['code'],
        padding='max_length',
        truncation=True,
        max_length=256
    )

train_dataset = train_dataset.map(tokenize_code, batched=True)
valid_dataset = valid_dataset.map(tokenize_code, batched=True)

# Use Data Collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# Define TrainingArguments and Trainer
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=10_000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
def save_model_and_tokenizer(model, tokenizer, path="/content/gdrive/MyDrive/final/fine_tuned_model_final"):
    # Save model
    model.save_pretrained(path)

    # Save tokenizer
    tokenizer.save_pretrained(path)

    print(f"Model and tokenizer saved to {path}")

save_model_and_tokenizer(model, tokenizer, path="/content/gdrive/MyDrive/final/fine_tuned_model_final")

model_path = "/content/gdrive/MyDrive/final/fine_tuned_model_final"

# Load the fine-tuned model and tokenizer
loaded_model = BertForSequenceClassification.from_pretrained(model_path)
loaded_tokenizer = BertTokenizer.from_pretrained(model_path)

print("Model and tokenizer loaded successfully!")

# Replace `outputs.last_hidden_state[:, 0, :]` with a mean pooling for simplicity
def get_batch_embeddings(snippets):
    inputs = loaded_tokenizer(snippets, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = loaded_model.bert(**inputs)
    # Mean pooling over token embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

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
train_dataset = load_dataset('json', data_files='/content/gdrive/MyDrive/final/train_dataset.json')['train']
test_dataset = load_dataset('json', data_files='/content/gdrive/MyDrive/final/test_dataset.json')['train']
valid_dataset = load_dataset('json', data_files='/content/gdrive/MyDrive/final/valid_dataset.json')['train']

# Step 3: Shuffle and select 100 random samples
train_dataset = train_dataset.shuffle(seed=42).select(range(4000))
valid_dataset = valid_dataset.shuffle(seed=42).select(range(500))
test_dataset = test_dataset.shuffle(seed=42).select(range(500))

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

# Step 2: Embedding generation for similarity retrieval
# Load candidate snippets from candidates.json
candidate_dataset = load_dataset('json', data_files='/content/gdrive/MyDrive/final/candidates.json')['train']
candidate_snippets = [example['code'] for example in candidate_dataset]

for i in range(0, len(candidate_snippets), batch_size):
    batch = candidate_snippets[i:i + batch_size]
    batch_embeddings = get_batch_embeddings(batch)
    candidate_embeddings.extend(batch_embeddings)

# Convert embeddings to a NumPy array for cosine similarity calculation
candidate_embeddings = np.array(candidate_embeddings)

# Function to retrieve top K semantically similar snippets
def retrieve_top_k_similar(query_code, K):
    query_embedding = get_batch_embeddings([query_code])[0]  # Get embedding for query
    similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]  # Compute similarity

    # Get indices of top K similar snippets
    top_k_indices = np.argsort(similarities)[-K:][::-1]
    top_k_snippets = [candidate_snippets[i] for i in top_k_indices]
    top_k_scores = [similarities[i] for i in top_k_indices]

    return top_k_snippets, top_k_scores

# Example usage with a query code snippet
query_code = """// Sample query code to find top-K similar snippets
int sumArray(int arr[], int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}"""
K = 5

top_k_snippets, top_k_scores = retrieve_top_k_similar(query_code, K)

# Print results
print("Top K Similar Snippets:")
for i, snippet in enumerate(top_k_snippets):
    print(f"\nSnippet {i + 1}:")
    print(snippet)
    print(f"Similarity Score: {top_k_scores[i]:.4f}")