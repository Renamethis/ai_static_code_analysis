from huggingface_hub import login
import sys
from datasets import load_dataset
from transformers import AutoTokenizer

# Load your CSV dataset
# The dataset should have two columns: "vulnerable_code" and "fixed_code"
dataset = load_dataset("json", data_files="dataset_tuned.json")
# Split the dataset into training and validation sets
dataset = dataset["train"].train_test_split(test_size=0.2)
print('test')
# Initialize the tokenizer for Llama
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")#"TinyPixel/Llama-2-7B-bf16-sharded")  # Replace with your desired Llama model
print('test2')
# Tokenize the data
def tokenize_function(example):
    print(len(example))
    # Prepare input as "Fix the vulnerability: [vulnerable_code]" and target as "fixed_code"
    example["input_text"] = f"Fix the vulnerability: {example['func_before']}"
    example["target_text"] = example["vul_func_with_fix"]
    
    # Tokenize input and target
    input_encodings = tokenizer(example["input_text"], truncation=True, max_length=512)
    target_encodings = tokenizer(example["target_text"], truncation=True, max_length=512)

    # Return the encodings as input_ids and labels
    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"]
    }
# Apply tokenization to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=10, writer_batch_size=10)
print('test3')
# Set the dataset format for PyTorch
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

# Load the pre-trained Llama model for fine-tuning
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./llama-vuln-fixer",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Train the model
trainer.train()
