from datasets import load_dataset
from transformers import RobertaTokenizer

# Load dataset from jsonl
dataset = load_dataset('json', data_files='train.jsonl')

# Load tokenizer (CodeBERT or GraphCodeBERT)
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Tokenize dataset
def tokenize_code(example):
    return tokenizer(example['code'], padding='max_length', truncation=True, max_length=5000)

tokenized_dataset = dataset.map(tokenize_code, batched=True)

print(tokenized_dataset['train'])