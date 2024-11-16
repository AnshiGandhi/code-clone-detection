# from datasets import load_dataset
# from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# # Load dataset from jsonl
# train_dataset = load_dataset('json', data_files='train.jsonl', split='train')
# valid_dataset = load_dataset('json', data_files='valid.jsonl', split='train')

# # Load tokenizer (CodeBERT or GraphCodeBERT)
# tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# # Tokenize dataset (adjust max_length to 512)
# def tokenize_code(example):
#     return tokenizer(example['code'], padding='max_length', truncation=True, max_length=512)

# train_dataset = train_dataset.map(tokenize_code, batched=True)
# valid_dataset = valid_dataset.map(tokenize_code, batched=True)

# # Convert labels to integers
# def process_labels(example):
#     example['label'] = int(example['label'])  # Convert label to an integer
#     return example

# # Apply the function to both train and validation datasets
# train_dataset = train_dataset.map(process_labels)
# valid_dataset = valid_dataset.map(process_labels)

# # Load the model
# codebert_model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=65)

# # Training setup
# training_args = TrainingArguments(
#     output_dir='./results',
#     evaluation_strategy="epoch",
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     logging_dir='./logs',
#     logging_steps=10,
# )

# # Trainer setup
# trainer = Trainer(
#     model=codebert_model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=valid_dataset
# )

# # Fine-tuning
# trainer.train()

# # Save the model and tokenizer
# model_save_path = './fine_tuned_robertamodel'
# codebert_model.save_pretrained(model_save_path)
# tokenizer.save_pretrained(model_save_path)








# This is for Roberta Model

import numpy as np
import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load datasets for fine-tuning
train_dataset = load_dataset('json', data_files='train_copy.jsonl')['train']
valid_dataset = load_dataset('json', data_files='valid_copy.jsonl')['train']
test_dataset = load_dataset('json', data_files='test_copy.jsonl')['train']

def process_labels(example):
    example['label'] = int(example['label'])  # Convert label from string to integer
    return example

train_dataset = train_dataset.map(process_labels)
valid_dataset = valid_dataset.map(process_labels)
test_dataset = test_dataset.map(process_labels)

# Load RoBERTa tokenizer and pre-trained RoBERTa model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=81)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenize datasets
def tokenize_code(example):
    return tokenizer(
        example['code'], 
        padding='max_length', 
        truncation=True, 
        max_length=512
    )

train_dataset = train_dataset.map(tokenize_code, batched=True)
valid_dataset = valid_dataset.map(tokenize_code, batched=True)

# Define TrainingArguments and Trainer
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=10_000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)

# Fine-tune the model
trainer.train()

# Step 2: Embedding generation for similarity retrieval
# Load candidate snippets from candidates.json
candidate_dataset = load_dataset('json', data_files='candidates.json')['train']
candidate_snippets = [example['code'] for example in candidate_dataset]

# Function to compute embeddings for a batch of code snippets
def get_batch_embeddings(snippets):
    inputs = tokenizer(
        snippets, return_tensors='pt', 
        padding=True, truncation=True, max_length=512
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model.roberta(**inputs)  # Directly access RoBERTa part of model
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token embeddings

# Generate embeddings for candidate snippets in batches
batch_size = 16
candidate_embeddings = []

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
