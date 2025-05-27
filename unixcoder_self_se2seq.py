import torch
import torch.nn as nn
from unixcoder import UniXcoder
from torch.utils.data import DataLoader
from datasets import load_dataset
import sacrebleu
from tqdm import tqdm
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and encoder
tokenizer = AutoTokenizer.from_pretrained("microsoft/UniXcoder-base")
encoder = AutoModel.from_pretrained("microsoft/UniXcoder-base").to(device)

# Hyperparameters
MAX_LENGTH = 256
BATCH_SIZE = 8  # Increased batch size for better gradient stability
EPOCHS = 1
LEARNING_RATE = 2e-5  # Slightly lower learning rate for stability
HIDDEN_SIZE = encoder.config.hidden_size
VOCAB_SIZE = tokenizer.vocab_size
NUM_LAYERS = 6
NUM_HEADS = 8
BEAM_SIZE = 8  # Reduced beam size for faster evaluation, can be tuned
DROPOUT = 0.1  # Added dropout for regularization

def tokenize(examples):
    # Simply return the text, tokenization will be done in collatefn
    return {
        "buggy": examples["buggy"],
        "fixed": examples["fixed"]
    }

def collate_fn(batch):
    # Tokenize here to avoid double tokenization
    buggytexts = [item["buggy"] for item in batch]
    fixedtexts = [item["fixed"] for item in batch]
    
    # Tokenize source
    source_encoding = tokenizer(
        buggytexts,
        maxlength=256,
        truncation=True,
        padding=True,
        returntensors="pt"
    )
    
    # Tokenize target
    target_encoding = tokenizer(
        fixedtexts,
        maxlength=256,
        truncation=True,
        padding=True,
        returntensors="pt"
    )
    
    # Prepare decoder input and labels
    decoder_input_ids = target_encoding["input_ids"][:, :-1]
    labels = target_encoding["input_ids"][:, 1:].clone()
    
    # Replace padding token id with -100 for loss calculation
    labels[labels == tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": source_encoding["input_ids"],
        "attention_mask": source_encoding["attention_mask"],
        "decoder_input_ids": decoder_input_ids,
        "labels": labels
    }

def beam_search(decoder, memory, start_token_id, end_token_id, beam_size=BEAM_SIZE, max_length=100):
    batch_size = memory.size(0)  # Dynamically set batch size
    device = memory.device
    
    beams = torch.full((batch_size, beam_size, 1), start_token_id, dtype=torch.long, device=device)
    scores = torch.zeros(batch_size, beam_size, device=device)
    active_beams = torch.ones(batch_size, beam_size, dtype=torch.bool, device=device)
    
    for step in range(max_length - 1):
        if step > max_length - 2:
            print(f"Warning: Reached max length {max_length}")
            break
        decoder_input = beams.view(batch_size * beam_size, -1)
        memory_expanded = memory.repeat_interleave(beam_size, dim=0)
        logits = decoder(decoder_input, memory_expanded)[:, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs = log_probs.view(batch_size, beam_size, -1)
        log_probs = log_probs.masked_fill(~active_beams.unsqueeze(-1), float('-inf'))
        
        new_scores = scores.unsqueeze(-1) + log_probs
        new_scores_2d = new_scores.view(batch_size, beam_size * VOCAB_SIZE)
        top_scores, top_indices = torch.topk(new_scores_2d, k=beam_size, dim=-1)
        
        new_beam_indices = top_indices // VOCAB_SIZE
        new_token_indices = top_indices % VOCAB_SIZE
        
        beams_expanded = beams.unsqueeze(2).repeat(1, 1, VOCAB_SIZE, 1).view(batch_size, beam_size * VOCAB_SIZE, -1)
        batch_indices = torch.arange(batch_size, device=device).view(-1, 1).repeat(1, beam_size)
        selected_beams = beams_expanded[batch_indices, top_indices, :]
        
        new_tokens = new_token_indices.unsqueeze(-1)
        beams = torch.cat([selected_beams, new_tokens], dim=-1)
        
        scores = top_scores
        
        active_beams = active_beams & (new_token_indices != end_token_id)
        
        if active_beams.sum(dim=-1).eq(0).all():
            break
    
    best_beam_idx = scores.argmax(dim=-1)
    batch_indices = torch.arange(batch_size, device=device)
    best_beams = beams[batch_indices, best_beam_idx, :]
    
    return best_beams
def compute_bleu(preds, targets):
        preds = [pred.strip() for pred in preds]
        targets = [target.strip() for target in targets]
        return sacrebleu.corpus_bleu(preds, [targets]).score

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class UniXCoderSeq2Seq(nn.Module):
    def __init__(self, hidden_size=768, num_decoder_layers=6, num_heads=12):
        super().__init__()
        self.encoder = unixcoder_model
        self.hidden_size = hidden_size
        
        # Build custom decoder
        self.decoder_embedding = nn.Embedding(tokenizer.vocab_size, hidden_size)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_size)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, tokenizer.vocab_size)
        
        # Copy encoder embeddings to decoder
        self.decoder_embedding.weight = self.encoder.embeddings.word_embeddings.weight
        
    def encode(self, input_ids, attention_mask):
        # Use UniXCoder as encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        
        return encoder_outputs
    
    def decode(self, tgt_input_ids, encoder_outputs, src_attention_mask, tgt_attention_mask=None):
        # Embed target tokens
        tgt_embeddings = self.decoder_embedding(tgt_input_ids)
        tgt_embeddings = self.positional_encoding(tgt_embeddings)
        
        # Create masks
        tgt_seq_len = tgt_input_ids.size(1)
        if tgt_attention_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_seq_len, 
                device=tgt_input_ids.device
            )
        else:
            tgt_mask = tgt_attention_mask
        
        # Invert attention mask for memory_key_padding_mask
        memory_key_padding_mask = ~src_attention_mask.bool()
        
        # Decode
        decoder_outputs = self.decoder(
            tgt_embeddings,
            encoder_outputs,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(decoder_outputs)
        
        return logits
    
    def forward(self, input_ids, attention_mask, decoder_input_ids):
        # Encode
        encoder_outputs = self.encode(input_ids, attention_mask)
        
        # Decode
        logits = self.decode(decoder_input_ids, encoder_outputs, attention_mask)
        
        return logits

from transformers import get_linear_schedule_with_warmup

config = RobertaConfig.from_pretrained("microsoft/UniXcoder-base")
decoder = TransformerDecoder(HIDDEN_SIZE, VOCAB_SIZE, NUM_LAYERS, config.num_attention_heads).to(device)
model = UniXCoderSeq2Seq(encoder, decoder).to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

train_dataset = load_dataset("code_x_glue_cc_code_refinement", "small", split="train")
val_dataset = load_dataset("code_x_glue_cc_code_refinement", "small", split="validation[:10]")
tokenized_train = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
tokenized_val = val_dataset.map(tokenize, remove_columns=val_dataset.column_names)
train_loader = DataLoader(tokenized_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(tokenized_val, batch_size=32, collate_fn=collate_fn)

# Learning rate scheduler
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=True)
    for batch in progress_bar:
        src_input_ids = batch["src_input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tgt_input_ids = batch["tgt_input_ids"][:, :-1].to(device)
        labels = batch["labels"][:, 1:].to(device)

        outputs = model(src_input_ids, tgt_input_ids, attention_mask)
        loss = criterion(outputs.view(-1, VOCAB_SIZE), labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()  # Update learning rate
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
    # Evaluation
    model.eval()
    predictions, references = [], []
    
    print(f"Starting evaluation on {len(valid_loader)} batches...")
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(valid_loader, desc="Evaluating", leave=False)):
            try:
                src_input_ids = batch["src_input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                memory = encoder(input_ids=src_input_ids, attention_mask=attention_mask).last_hidden_state

                start_token_id = tokenizer.cls_token_id
                end_token_id = tokenizer.sep_token_id

                decoder_input = torch.full((src_input_ids.size(0), 1), start_token_id, device=device)
                
                # Add timeout or max steps to beam search
                best_beams = beam_search(decoder, memory, start_token_id, end_token_id, max_length=100)
                
                decoded_preds = tokenizer.batch_decode(best_beams, skip_special_tokens=True)
                decoded_refs = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
                predictions.extend(decoded_preds)
                references.extend(decoded_refs)
                
                # Limit evaluation to first few batches for debugging
                if idx >= 10:  # Only evaluate on 10 batches for now
                    print("Limited evaluation for debugging...")
                    break
                    
            except Exception as e:
                print(f"Error in batch {idx}: {e}")
                continue

    bleu_score = compute_bleu(predictions, references) if predictions else 0.0
    print(f"Validation BLEU score: {bleu_score:.2f}")


