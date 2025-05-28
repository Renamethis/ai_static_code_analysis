import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, 
    RobertaModel,
    EncoderDecoderModel,
    EncoderDecoderConfig,
    get_linear_schedule_with_warmup,
    AdamW
)
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import evaluate
from torch.cuda.amp import GradScaler, autocast
import json
import os

class CodeRefinementDataset(Dataset):
    def __init__(self, data, tokenizer, max_source_length=256, max_target_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process source (buggy code)
        source_text = item['old_comment'] + " " + item['old_code']
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Process target (refined code)
        target_text = item['new_comment'] + " " + item['new_code']
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels (shift tokens for autoregressive generation)
        labels = target_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'decoder_input_ids': target_encoding['input_ids'].squeeze(),
            'decoder_attention_mask': target_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

class UniXcoderForCodeRefinement:
    def init(self, modelname="microsoft/unixcoder-base", device='cuda'):
        self.device = device
        self.tokenizer = RobertaTokenizer.frompretrained(modelname)
        
        # Initialize encoder-decoder model
        self.model = self.createencoderdecodermodel(modelname)
        self.model.to(device)
        
        # Add special tokens if needed
        specialtokens = ['<mask0>', '<mask1>', '<mask2>']
        self.tokenizer.addspecialtokens({'additionalspecialtokens': specialtokens})
        self.model.resizetokenembeddings(len(self.tokenizer))
        
    def createencoderdecodermodel(self, modelname):
        """Create encoder-decoder model from UniXcoder"""
        # Load UniXcoder as encoder
        encoder = RobertaModel.frompretrained(modelname)
        
        # Create encoder-decoder configuration
        config = EncoderDecoderConfig.fromencoderdecoderconfigs(
            encoder.config, 
            encoder.config
        )
        
        # Important configurations for better performance
        config.decoderstarttokenid = self.tokenizer.clstokenid
        config.padtokenid = self.tokenizer.padtokenid
        config.eostokenid = self.tokenizer.septokenid
        config.bostokenid = self.tokenizer.clstokenid
        
        # Create model with shared weights between encoder and decoder
        model = EncoderDecoderModel(config=config)
        model.encoder = encoder
        model.decoder = RobertaModel.frompretrained(modelname)
        
        # Initialize cross-attention layers properly
        model.decoder.config.isdecoder = True
        model.decoder.config.addcrossattention = True
        
        return model
    
    def train(self, traindataset, valdataset, config):
        """Training function with best practices"""
        # Create data loaders
        trainloader = DataLoader(
            traindataset, 
            batchsize=config['batchsize'], 
            shuffle=True,
            numworkers=4,
            pinmemory=True
        )
        
        valloader = DataLoader(
            valdataset, 
            batchsize=config['batchsize'], 
            shuffle=False,
            numworkers=4,
            pinmemory=True
        )
        
        # Optimizer with different learning rates for different parts
        optimizergroupedparameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if "decoder" in n and p.requires_grad],
                "lr": config'decoder_lr',
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if "decoder" not in n and p.requires_grad],
                "lr": config'encoder_lr',
            }
        ]
# Load dataset
dataset = load_dataset("code_x_glue_cc_code_refinement", "small")

# Initialize tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "microsoft/unixcoder-base", 
    "microsoft/unixcoder-base"
)

# Configure model
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.sep_token_id

# Preprocessing function
def preprocess_function(examples):
    inputs = [ex['old_comment'] + " " + ex['old_code'] for ex in examples['old']]
    targets = [ex['new_comment'] + " " + ex['new_code'] for ex in examples['new']]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=256, 
        truncation=True, 
        padding='max_length'
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, 
            max_length=256, 
            truncation=True, 
            padding='max_length'
        )
    
    # Replace padding token id's of the labels by -100 so it's ignored by loss
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] 
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Process datasets
train_dataset = dataset['train'].map(
    preprocess_function, 
    batched=True, 
    remove_columns=dataset['train'].column_names
)
val_dataset = dataset['validation'].map(
    preprocess_function, 
    batched=True, 
    remove_columns=dataset['validation'].column_names
)

# Convert to torch format
train_dataset.set_format('torch')
val_dataset.set_format('torch')

# Training configuration
config = {
    'batch_size': 16,
    'gradient_accumulation_steps': 4,
    'learning_rate': 5e-5,
    'warmup_steps': 1000,
    'num_epochs': 1,
    'max_grad_norm': 1.0,
    'fp16': True,
    'eval_steps': 500,
    'save_steps': 1000,
    'logging_steps': 50
}

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=config['batch_size'], 
    shuffle=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=config['batch_size'], 
    shuffle=False
)

# Setup training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
total_steps = len(train_loader) * config['num_epochs'] // config['gradient_accumulation_steps']
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=config['warmup_steps'],
    num_training_steps=total_steps
)

scaler = GradScaler() if config['fp16'] else None
bleu = evaluate.load("bleu")

# Training loop
global_step = 0
best_bleu_score = 0

for epoch in range(config['num_epochs']):
    # Training
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass with mixed precision
        if config['fp16']:
            with autocast():
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss / config['gradient_accumulation_steps']
        else:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss / config['gradient_accumulation_steps']
        
        # Backward pass
        if config['fp16']:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        total_loss += loss.item()
        
        # Update weights
        if (step + 1) % config['gradient_accumulation_steps'] == 0:
            if config['fp16']:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / (step + 1),
                'lr': scheduler.get_last_lr()[0]
            })
            
            # Evaluation
            if global_step % config['eval_steps'] == 0:
                print(f"\nEvaluating at step {global_step}...")
                model.eval()
                eval_loss = 0
                predictions = []
                references = []
                
    with torch.no_grad():
        for eval_batch in tqdm(val_loader, desc="Evaluating"):
            eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
            
            # Calculate validation loss
            outputs = model(
                input_ids=eval_batch['input_ids'],
                attention_mask=eval_batch['attention_mask'],
                labels=eval_batch['labels']
            )
            eval_loss += outputs.loss.item()
            
            # Generate predictions
            generated_ids = model.generate(
                input_ids=eval_batch['input_ids'],
                attention_mask=eval_batch['attention_mask'],
                max_length=256,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                temperature=0.8,
                do_sample=False,
                repetition_penalty=1.2
            )
            
            # Decode predictions
            preds = tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Process labels for references - FIX HERE
            labels = eval_batch['labels'].clone()
            # Replace -100 with pad token id before decoding
            labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
            
            # Ensure labels are on CPU and converted to list for batch_decode
            refs = tokenizer.batch_decode(
                labels.cpu().tolist(), 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            predictions.extend(preds)
            references.extend([[ref] for ref in refs])

    # Calculate metrics
    avg_eval_loss = eval_loss / len(val_loader)
    bleu_score = bleu.compute(
        predictions=predictions, 
        references=references,
        max_order=4,
        smooth=True
    )

    # Additional metrics
    exact_match = sum([pred.strip() == ref[0].strip() for pred, ref in zip(predictions, references)]) / len(predictions)

    print(f"Step {global_step}: Eval Loss: {avg_eval_loss:.4f}, BLEU: {bleu_score['bleu']:.4f}, Exact Match: {exact_match:.4f}")
    print(f"BLEU-1: {bleu_score['precisions'][0]:.4f}, BLEU-2: {bleu_score['precisions'][1]:.4f}, "
        f"BLEU-3: {bleu_score['precisions'][2]:.4f}, BLEU-4: {bleu_score['precisions'][3]:.4f}")

    
    # End of epoch evaluation
    print(f"\nEnd of Epoch {epoch+1} Evaluation...")
    model.eval()
    evalloss = 0
    predictions = []
    references = []
    
    with torch.nograd():
        for evalbatch in tqdm(valloader, desc="Final Evaluation"):
            evalbatch = {k: v.to(device) for k, v in evalbatch.items()}
            
            outputs = model(
                inputids=evalbatch'input_ids',
                attentionmask=evalbatch'attention_mask',
                labels=evalbatch['labels']
            )
            evalloss += outputs.loss.item()
            
            generatedids = model.generate(
                inputids=evalbatch['inputids'],
                attentionmask=evalbatch'attention_mask',
                maxlength=256,
                numbeams=5,
                earlystopping=True,
                norepeatngramsize=3,
                lengthpenalty=1.0
            )
            
            preds = tokenizer.batchdecode(generatedids, skipspecialtokens=True)
            labels = evalbatch'labels'
            labelslabels == -100 = tokenizer.padtoken