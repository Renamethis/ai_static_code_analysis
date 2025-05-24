from datasets import load_dataset

# This loads the "small" dataset (CodeXGLUE - code refinement)
dataset = load_dataset("code_x_glue_cc_code_refinement", "small")
train_data = dataset["train"]
valid_data = dataset["validation"]
test_data = dataset["test"]

from transformers import AutoTokenizer, AutoModelForCausalLM

model_checkpoint = "codellama/CodeLlama-7b-hf"  # Or smaller model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

def preprocess(example, max_input_length=512, max_target_length=256):
    # Add prompt-style prefix for clarity
    input_text = f"<s>Fix the following buggy code:\n{example['buggy']}\n</s>"
    target_text = example["fixed"] + tokenizer.eos_token
    
    input_ids = tokenizer.encode(input_text, max_length=max_input_length, truncation=True)
    target_ids = tokenizer.encode(target_text, max_length=max_target_length, truncation=True)
    
    return {
        "input_ids": input_ids,
        "labels": target_ids
    }

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
    fp16=True,  # if using GPU with fp16
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

