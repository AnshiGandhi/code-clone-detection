from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CloneDetector:
    def __init__(self, model_path):
        self.batch_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_model = BertForSequenceClassification.from_pretrained(model_path)
        self.loaded_tokenizer = BertTokenizer.from_pretrained(model_path)
        self.candidate_embeddings = []

    def get_batch_embeddings(self, batch):
        # Tokenize and move to device
        inputs = self.loaded_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Get outputs from the base BERT model
            outputs = self.loaded_model.bert(**inputs)  # Access the underlying BERT model
        # Return CLS token embeddings
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    def retrieve_top_k_similar(self, query_code, keys, K):
        query_embedding = self.get_batch_embeddings([query_code])[0]  # Get embedding for query
        similarities = cosine_similarity([query_embedding], self.candidate_embeddings)[0]  # Compute similarity

        # Get indices of top K similar snippets
        top_k_indices = np.argsort(similarities)[-K:][::-1]
        # top_k_snippets = [keys[i] for i in top_k_indices]
        # top_k_scores = [similarities[i] for i in top_k_indices]

        top_k_result = {
            keys[i]: similarities[i]
            for i in top_k_indices
        }

        return top_k_result

    def process_candidates(self, keys, candidates, k, code):
        """
        Process the candidates dictionary and return the first 'k' items as a result.

        Args:
            candidates (dict): A dictionary of candidates.
            k (int): Number of items to return.

        Returns:
            dict: Processed result containing 'k' key-value pairs.
        """
        # Validate candidates input
        if not isinstance(candidates, list):
            raise ValueError("Candidates must be a list.")
        
        if not isinstance(keys, list):
            raise ValueError("Keys must be a list.")
        
        if not isinstance(code, str):
            raise ValueError("Code must be a string.")
        
        if not isinstance(k, int):
            raise ValueError("k must be an integer.")

        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i:i + self.batch_size]
            batch_embeddings = self.get_batch_embeddings(batch)
            self.candidate_embeddings.extend(batch_embeddings)

        # Convert embeddings to a NumPy array for cosine similarity calculation
        self.candidate_embeddings = np.array(self.candidate_embeddings)

        return self.retrieve_top_k_similar(code, keys, k)