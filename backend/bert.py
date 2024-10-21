from datasets import load_dataset
from transformers import BertTokenizer

# Load dataset from jsonl file
dataset = load_dataset('json', data_files='train.jsonl')

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
def tokenize_code(example):
    # Tokenize the 'code' field with padding and truncation
    return tokenizer(
        example['code'], 
        padding='max_length', 
        truncation=True, 
        max_length=512  # Adjusted for BERT's limit
    )

# Apply the tokenizer on the entire dataset
tokenized_dataset = dataset.map(tokenize_code, batched=True)
