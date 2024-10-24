from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# Load dataset from jsonl
train_dataset = load_dataset('json', data_files='train.jsonl', split='train')
valid_dataset = load_dataset('json', data_files='valid.jsonl', split='train')

# Load tokenizer (CodeBERT or GraphCodeBERT)
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Tokenize dataset (adjust max_length to 512)
def tokenize_code(example):
    return tokenizer(example['code'], padding='max_length', truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_code, batched=True)
valid_dataset = valid_dataset.map(tokenize_code, batched=True)

# Convert labels to integers
def process_labels(example):
    example['label'] = int(example['label'])  # Convert label to an integer
    return example

# Apply the function to both train and validation datasets
train_dataset = train_dataset.map(process_labels)
valid_dataset = valid_dataset.map(process_labels)

# Load the model
codebert_model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=65)

# Training setup
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer setup
trainer = Trainer(
    model=codebert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)

# Fine-tuning
trainer.train()

# Save the model and tokenizer
model_save_path = './fine_tuned_robertamodel'
codebert_model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
