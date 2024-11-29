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
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



def get_batch_embeddings(batch):
    # Tokenize and move to device
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        # Get outputs from the base BERT model
        outputs = model.bert(**inputs)  # Access the underlying BERT model
    # Return CLS token embeddings
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


# Step 2: Embedding generation for similarity retrieval
# Load candidate snippets from candidates.json

batch_size = 32
candidate_embeddings = []
candidate_dataset = load_dataset('json', data_files='final/candidates.json')['train']
candidate_snippets = [example['key'] for example in candidate_dataset]

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