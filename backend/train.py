import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.model_selection import train_test_split

# Step 1: Load datasets for fine-tuning
train_dataset = load_dataset('json', data_files='train.jsonl')['train']

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
train_dataset.to_json('final/train_dataset.json')
valid_dataset.to_json('final/valid_dataset.json')
test_dataset.to_json('final/test_dataset.json')

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
def save_model_and_tokenizer(model, tokenizer, path="final/fine_tuned_model_final"):
    # Save model
    model.save_pretrained(path)

    # Save tokenizer
    tokenizer.save_pretrained(path)

    print(f"Model and tokenizer saved to {path}")

save_model_and_tokenizer(model, tokenizer, path="final/fine_tuned_model_final")


