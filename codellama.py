from datasets import load_dataset
import torch
torch.cuda.empty_cache()

# This loads the "small" dataset (CodeXGLUE - code refinement)
train_data = load_dataset("code_x_glue_cc_code_refinement", "small", split="train[:1%]")
valid_data = load_dataset("code_x_glue_cc_code_refinement", "small", split="validation[:1%]")
test_data  = load_dataset("code_x_glue_cc_code_refinement", "small", split="test[:1%]")
from transformers import AutoTokenizer, AutoModelForCausalLM

model_checkpoint = "codellama/CodeLlama-7b-hf"  # Or smaller model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
model.gradient_checkpointing_enable()
#  Fix: set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

def preprocess(example, max_length=64):
    prompt = f"Fix the following buggy code:\n{example['buggy']}\n"

    full_input = prompt + example["fixed"]  # Input + target together
    tokenized = tokenizer(
        full_input,
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        example["fixed"],
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )["input_ids"]

    # Replace padding token ids with -100 to ignore them in loss
    labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]

    tokenized["labels"] = labels
    return tokenized

tokenized_train = train_data.map(preprocess)
tokenized_valid = valid_data.map(preprocess)
tokenized_test = test_data.map(preprocess)

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

import evaluate
bleu = evaluate.load("bleu")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    return bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./codellama-code-refinement",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    bf16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate(eval_dataset=tokenized_test)

