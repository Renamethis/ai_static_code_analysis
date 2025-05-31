import torch
from torch import nn
from transformers import (
    RobertaModel,
    RobertaTokenizer,
    GPT2Model,
    GPT2Config,
    EncoderDecoderModel,
    EncoderDecoderConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    GPT2LMHeadModel
)
from datasets import load_dataset
import numpy as np
import evaluate
import os

# Disable SDPA for ROCm compatibility
os.environ["TORCH_SDPA_FLASH_ATTENTION_DISABLE"] = "1"

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")

# Add special tokens for code generation
special_tokens = {
    "additional_special_tokens": ["<java>", "<python>", "<go>", "<javascript>", "<ruby>", "<php>"]
}
tokenizer.add_special_tokens(special_tokens)

def create_test():
    from transformers import EncoderDecoderModel, RobertaTokenizer

    # This method properly initializes cross-attention layers
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        "microsoft/unixcoder-base",
        "gpt2"
    )

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure generation
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    return model

# Option 1: UniXCoder + GPT2 decoder
def create_unixcoder_gpt2_model():
    # Load UniXCoder as encoder
    encoder = RobertaModel.from_pretrained("microsoft/unixcoder-base")
    encoder.resize_token_embeddings(len(tokenizer))

    # Configure GPT-2 decoder with cross-attention
    decoder_config = GPT2Config.from_pretrained("gpt2")
    decoder_config.add_cross_attention = True  # This is the key fix
    decoder_config.is_decoder = True

    # Load decoder with the modified config
    decoder = GPT2LMHeadModel.from_pretrained("gpt2", config=decoder_config)

    # Create encoder-decoder model
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    # Configure the model
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    
    return model

# Option 2: UniXCoder encoder-decoder (both RoBERTa)
def create_unixcoder_encoder_decoder():
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        "microsoft/unixcoder-base",
        "microsoft/unixcoder-base"
    )
    
    # Resize embeddings for special tokens
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.decoder.resize_token_embeddings(len(tokenizer))
    
    # Configure for generation
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 10
    
    return model

# Choose model variant
print("Creating model...")
model = create_unixcoder_encoder_decoder()  # or create_unixcoder_encoder_decoder()

# Load CodeXGLUE dataset (example: Java to Python translation)
print("Loading dataset...")
dataset = {
    "train": load_dataset("code_x_glue_cc_code_refinement", "small", split="train"),
    "validation": load_dataset("code_x_glue_cc_code_refinement", "small", split="validation[:15%]"),
}

# Alternative: Load from local files if you have them
# from datasets import Dataset
# train_data = {"source": [...], "target": [...]}
# train_dataset = Dataset.from_dict(train_data)
# Preprocessing function
def preprocessfunction(examples):
    # Add language tags
    sources = examples["buggy"]
    targets = examples["fixed"]
    
    # Tokenize inputs
    modelinputs = tokenizer(
        sources,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    
    # Replace padding token id with -100
    labels["input_ids"] = [
        [(token_id if token_id != tokenizer.pad_token_id else -100) for token_id in label_sequence]
        for label_sequence in labels["input_ids"]
    ]

    
    modelinputs["labels"] = labels["input_ids"]
    
    # Debug first example
    
    return modelinputs

# Process datasets
print("Processing datasets...")
traindataset = dataset["train"].map(
    preprocessfunction,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Processing train dataset"
)

evaldataset = dataset["validation"].map(
    preprocessfunction,
    batched=True,
    remove_columns=dataset["validation"].column_names,
    desc="Processing validation dataset"
)

# Load BLEU metric
bleumetric = evaluate.load("bleu")

# Global variable to store sample predictions
evalsamples = []



def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    
    # Handle different prediction formats
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Decode labels (replace -100 with pad token id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Clean up
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # Print samples
    print("\n" + "="*80)
    print("EVALUATION SAMPLES:")
    print("="*80)
    for i in range(min(3, len(decoded_preds))):
        print(f"\n--- Sample {i+1} ---")
        print(f"TARGET: {decoded_labels[i][:200]}...")
        print(f"PREDICTION: {decoded_preds[i][:200]}...")
    print("="*80 + "\n")
    
    # Calculate BLEU
    bleu_score = bleumetric.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels]
    )
    
    return {"bleu": bleu_score["bleu"] * 100}

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./unixcoder-decoder-results",
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=50000,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    num_train_epochs=2,
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=10,
    fp16=True,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    warmup_steps=500,
    gradient_accumulation_steps=2,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=traindataset,
    eval_dataset=evaldataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Training
trainer.train()

# Save model
trainer.save_model("./final-model")
tokenizer.save_pretrained("./final-model")
