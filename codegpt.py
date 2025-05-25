from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import numpy as np
from evaluate import load
import torch

train = load_dataset("code_x_glue_cc_code_refinement", "medium", split="train[:1%]")
val = load_dataset("code_x_glue_cc_code_refinement", "medium", split="validation[:1%]")
# 2. Load tokenizer and model (CodeGPT small)
model_name = "microsoft/CodeGPT-small-py"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 3. Preprocessing function: concat buggy and fixed code as input-output pairs
def preprocess_function(examples):
    inputs = examples['buggy']
    targets = examples['fixed']

    # Tokenize inputs and targets
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')

    # We will train the model to predict the fixed code, so tokenize targets as labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 4. Tokenize dataset
tokenized_train = train.map(preprocess_function, batched=True)
tokenized_val = val.map(preprocess_function, batched=True)
# 5. Data collator for causal LM (mask padding tokens in labels)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6. Load BLEU metric
bleu_metric = load("bleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # If preds are logits, convert to token IDs
    if isinstance(preds, tuple):
        preds = preds[0]
    if preds.ndim == 3:  # shape: [batch_size, seq_len, vocab_size]
        preds = np.argmax(preds, axis=-1)

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in labels as those are ignored by loss
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ❗️Do NOT tokenize into words — BLEU can tokenize automatically
    bleu_score = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)

    return {"bleu": bleu_score["bleu"]}


# 8. Setup training arguments
training_args = TrainingArguments(
    output_dir="./codegpt-code-refinement",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_strategy="epoch",
    logging_steps=100,
    fp16=True,
)

training_args = TrainingArguments(
    output_dir="./codegpt-code-refinement",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_strategy="epoch",
    logging_steps=100,
    # REMOVE predict_with_generate here
)

# 9. Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 10. Train and evaluate
trainer.train()
generate_and_evaluate(model, tokenizer, tokenized_val)
