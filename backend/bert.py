from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset from jsonl file
#dataset = load_dataset('json', data_files='train.jsonl')

# Load BERT tokenizer
#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
# def tokenize_code(example):
#     # Tokenize the 'code' field with padding and truncation
#     return tokenizer(
#         example['code'], 
#         padding='max_length', 
#         truncation=True, 
#         max_length=512  # Adjusted for BERT's limit
#     )

# Apply the tokenizer on the entire dataset
# tokenized_dataset = dataset.map(tokenize_code, batched=True)


# Load the dataset from JSONL file
dataset = load_dataset('json', data_files='train.jsonl')['train']

# Set a limit to reduce dataset size if needed
LIMIT = 100  # Adjust based on available memory and time constraints
dataset = dataset.shuffle(seed=42).select(range(min(LIMIT, len(dataset))))

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Use GPU if available for faster processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenize the dataset
def tokenize_code(example):
    return tokenizer(
        example['code'], 
        padding='max_length', 
        truncation=True, 
        max_length=512
    )

# Apply tokenization to the dataset
tokenized_dataset = dataset.map(tokenize_code, batched=True)

# Extract original code snippets for later use
candidate_snippets = [example['code'] for example in dataset]

# Function to compute embeddings for a batch of code snippets
def get_batch_embeddings(snippets):
    inputs = tokenizer(
        snippets, return_tensors='pt', 
        padding=True, truncation=True, max_length=512
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token embeddings

# Generate embeddings for all candidate snippets in batches
batch_size = 16  # Adjust based on available memory
candidate_embeddings = []

for i in range(0, len(candidate_snippets), batch_size):
    batch = candidate_snippets[i:i + batch_size]
    batch_embeddings = get_batch_embeddings(batch)
    candidate_embeddings.extend(batch_embeddings)

# Convert embeddings to NumPy array for efficient computation
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

# Example usage
query_code = """int main(int argc, char* argv[]) {
    int shu[number];
    int n, i, j;
    int k = 0;
    scanf("%d", &shu[0]);
    for (n = 0; shu[n] != 0; n++) {
        scanf("%d", &shu[n + 1]);
    }
    for (i = 0; i <= n; i++) {
        for (j = 0; j <= n; j++) {
            if (shu[i] == 2 * shu[j]) {
                k++;
            }
        }
    }
    if (k != 0) {
        k = k - 1;
        printf("%d", k);
    } else printf("%d", k);
    return 0;
}"""  # Your query code snippet

K = 5  # Number of top similar snippets to retrieve

# Retrieve the top K similar code snippets
top_k_snippets, top_k_scores = retrieve_top_k_similar(query_code, K)

# Print the results
print("Top K Similar Snippets:")
for i, snippet in enumerate(top_k_snippets):
    print(f"\nSnippet {i + 1}:")
    print(snippet)
    print(f"Similarity Score: {top_k_scores[i]:.4f}")

