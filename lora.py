from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch

# Load dataset
train_data = load_dataset("code_x_glue_cc_code_refinement", "small", split="train[:1%]")
valid_data = load_dataset("code_x_glue_cc_code_refinement", "small", split="validation[:1%]")

# Load tokenizer
model_name = "codellama/CodeLlama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for k-bit LoRA
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Preprocessing
def preprocess(example, max_length=64):
    prompt = f"Fix the following buggy code:\n{example['buggy']}\n"
    input_ids = tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length)
    labels = tokenizer(example["fixed"], truncation=True, padding="max_length", max_length=max_length)["input_ids"]
    labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]
    input_ids["labels"] = labels
    return input_ids

tokenized_train = train_data.map(lambda x: preprocess(x), batched=True, remove_columns=train_data.column_names)
tokenized_valid = valid_data.map(lambda x: preprocess(x), batched=True, remove_columns=valid_data.column_names)

# Training args
training_args = TrainingArguments(
    output_dir="./codellama-lora-code-refine",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    logging_dir="./logs",
    save_total_limit=2,
    learning_rate=5e-5,
    bf16=True,
    report_to="none"
)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
