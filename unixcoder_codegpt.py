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
    EarlyStoppingCallback
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

# Option 1: UniXCoder + GPT2 decoder
def create_unixcoder_gpt2_model():
    # Load UniXCoder as encoder
    encoder = RobertaModel.from_pretrained("microsoft/unixcoder-base")
    encoder.resize_token_embeddings(len(tokenizer))
    
    # Create GPT2 decoder with matching hidden size
    decoder_config = GPT2Config(
        vocab_size=len(tokenizer),
        n_embd=768,  # Match UniXCoder hidden size
        n_layer=6,   # Smaller decoder
        n_head=12,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
    )
    decoder = GPT2Model(decoder_config)
    
    # Create encoder-decoder model
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    
    # Set special tokens
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    
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
    model.config.max_length = 512
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    
    return model

# Choose model variant
print("Creating model...")
model = create_unixcoder_gpt2_model()  # or create_unixcoder_encoder_decoder()

# Load CodeXGLUE dataset (example: Java to Python translation)
print("Loading dataset...")
dataset = load_dataset("code_x_glue_cc_code_refinement", "small")

# Alternative: Load from local files if you have them
# from datasets import Dataset
# train_data = {"source": [...], "target": [...]}
# train_dataset = Dataset.from_dict(train_data)
# Preprocessing function
def preprocessfunction(examples):
    # Add language tags
    sources = [f"<buggy> {code}" for code in examples["buggy"]]
    targets = [f"<fixed> {code}" for code in examples["fixed"]]
    
    # Tokenize inputs
    modelinputs = tokenizer(
        sources,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    
    # Replace padding token id with -100
    labels["input_ids"] = [
        [[(l if l != tokenizer.pad_token_id else -100) for l in labels]
        for label in labels["input_ids"]]
    ]
    
    modelinputs["labels"] = labels["input_ids"]
    
    # Debug first example
    # if len(sources) > 0:
    #     print(f"\nPreprocessing example:")
    #     print(f"Source: {sources}...")
    #     print(f"Target: {targets}...")
    
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

def create_unixcoder_gpt2_model():
    from transformers import GPT2LMHeadModel  # Use LMHead version instead of base
    
    # Load UniXCoder as encoder
    encoder = RobertaModel.from_pretrained("microsoft/unixcoder-base")
    encoder.resize_token_embeddings(len(tokenizer))
    
    # Create GPT2 decoder with matching hidden size and LM head
    decoder_config = GPT2Config(
        vocab_size=len(tokenizer),
        n_embd=768,  # Match UniXCoder hidden size
        n_layer=6,   # Smaller decoder
        n_head=12,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
    )
    decoder = GPT2LMHeadModel(decoder_config)  # Changed from GPT2Model
    
    # Create encoder-decoder model
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    
    # Set special tokens
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    
    return model



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
    bleu_score = bleu_metric.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels]
    )
    
    return {"bleu": bleu_score["bleu"] * 100}

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./unixcoder-decoder-results",
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    num_train_epochs=3,
    predict_with_generate=True,
    generation_max_length=512,
    generation_num_beams=4,
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
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Training
trainer.train()

# Save model
trainer.save_model("./final-model")
tokenizer.save_pretrained("./final-model")
