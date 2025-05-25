import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, RobertaConfig
from datasets import load_dataset
import sacrebleu
from tqdm import tqdm
import math

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and encoder
tokenizer = AutoTokenizer.from_pretrained("microsoft/UniXcoder-base")
encoder = AutoModel.from_pretrained("microsoft/UniXcoder-base").to(device)

# Hyperparameters
MAX_LENGTH = 256
BATCH_SIZE = 3
EPOCHS = 5
LEARNING_RATE = 5e-5
HIDDEN_SIZE = encoder.config.hidden_size
VOCAB_SIZE = tokenizer.vocab_size
NUM_LAYERS = 6
NUM_HEADS = 8
BEAM_SIZE = 10  # Number of beams for beam search

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_layers, num_heads, max_len=MAX_LENGTH):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt_input_ids, memory, tgt_mask=None, memory_mask=None):
        embedded = self.embedding(tgt_input_ids)
        embedded = embedded.transpose(0, 1)
        embedded = self.positional_encoding(embedded)
        output = self.transformer_decoder(embedded, memory.transpose(0, 1), tgt_mask=tgt_mask, memory_mask=memory_mask)
        output = self.fc_out(output.transpose(0, 1))
        return output

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_input_ids, tgt_input_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids=src_input_ids, attention_mask=attention_mask).last_hidden_state
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input_ids.size(1)).to(device)
        logits = self.decoder(tgt_input_ids, memory=encoder_outputs, tgt_mask=tgt_mask)
        return logits

# Tokenize function
def tokenize(example):
    source = tokenizer(
        example["buggy"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    target = tokenizer(
        example["fixed"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    return {
        "src_input_ids": source["input_ids"],
        "attention_mask": source["attention_mask"],
        "tgt_input_ids": target["input_ids"],
        "labels": target["input_ids"]
    }

def collate_fn(batch):
    return {
        key: torch.stack([torch.tensor(example[key], dtype=torch.long) for example in batch])
        for key in batch[0]
    }

# BLEU evaluation
def compute_bleu(preds, targets):
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]
    return sacrebleu.corpus_bleu(preds, [targets]).score

# Load dataset
train_dataset = load_dataset("code_x_glue_cc_code_refinement", "medium", split="train[:20%]")
val_dataset = load_dataset("code_x_glue_cc_code_refinement", "medium", split="validation[:5%]")
tokenized_train = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
tokenized_val = val_dataset.map(tokenize, remove_columns=val_dataset.column_names)
train_loader = DataLoader(tokenized_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(tokenized_val, batch_size=1, collate_fn=collate_fn)

RobertaConfig.from_pretrained("microsoft/UniXcoder-base")
# Model setup
decoder = TransformerDecoder(HIDDEN_SIZE, VOCAB_SIZE, NUM_LAYERS, NUM_HEADS).to(device)
model = Seq2Seq(encoder, decoder).to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# BLEU evaluation
def compute_bleu(preds, targets):
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]
    return sacrebleu.corpus_bleu(preds, [targets]).score

# Beam Search Implementation
def beam_search(decoder, memory, start_token_id, end_token_id, beam_size=BEAM_SIZE, max_length=MAX_LENGTH):
    batch_size = 1#memory.size(0)
    device = memory.device
    
    # Initialize beams: each batch item starts with beam_size copies of the start token
    beams = torch.full((batch_size, beam_size, 1), start_token_id, dtype=torch.long, device=device)
    # Initialize cumulative log probabilities for each beam
    scores = torch.zeros(batch_size, beam_size, device=device)
    # Track which beams are still active (not finished)
    active_beams = torch.ones(batch_size, beam_size, dtype=torch.bool, device=device)
    
    for step in range(max_length - 1):
        # Reshape beams for decoder input (batch_size * beam_size, seq_len)
        decoder_input = beams.view(batch_size * beam_size, -1)
        # Expand memory to match the beam size
        memory_expanded = memory.repeat_interleave(beam_size, dim=0)
        
        # Get logits for the next token
        logits = decoder(decoder_input, memory_expanded)[:, -1, :]  # Shape: (batch_size * beam_size, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)  # Convert to log probabilities
        
        # Reshape log_probs to (batch_size, beam_size, vocab_size)
        log_probs = log_probs.view(batch_size, beam_size, -1)
        
        # For inactive beams, set log_probs to a very low value to avoid selection
        log_probs = log_probs.masked_fill(~active_beams.unsqueeze(-1), float('-inf'))
        
        # Compute new scores by adding log probabilities to existing scores
        new_scores = scores.unsqueeze(-1) + log_probs  # Shape: (batch_size, beam_size, vocab_size)
        
        # Get top beam_size candidates for each batch item
        new_scores_2d = new_scores.view(batch_size, beam_size * VOCAB_SIZE)
        top_scores, top_indices = torch.topk(new_scores_2d, k=beam_size, dim=-1)
        
        # Extract beam index and token index from top_indices
        new_beam_indices = top_indices // VOCAB_SIZE  # Which beam the token came from
        new_token_indices = top_indices % VOCAB_SIZE  # Which token was selected
        
        # Gather the new beams based on the selected beam indices
        beams_expanded = beams.unsqueeze(2).repeat(1, 1, VOCAB_SIZE, 1).view(batch_size, beam_size * VOCAB_SIZE, -1)
        batch_indices = torch.arange(batch_size, device=device).view(-1, 1).repeat(1, beam_size)
        selected_beams = beams_expanded[batch_indices, top_indices, :]
        
        # Append the new tokens to the selected beams
        new_tokens = new_token_indices.unsqueeze(-1)  # Shape: (batch_size, beam_size, 1)
        beams = torch.cat([selected_beams, new_tokens], dim=-1)
        
        # Update scores
        scores = top_scores
        
        # Check for end token to mark beams as finished
        active_beams = active_beams & (new_token_indices != end_token_id)
        
        # If all beams in a batch are finished, stop
        if active_beams.sum(dim=-1).eq(0).all():
            break
    
    # Select the best beam for each batch item based on final scores
    best_beam_idx = scores.argmax(dim=-1)  # Shape: (batch_size,)
    batch_indices = torch.arange(batch_size, device=device)
    best_beams = beams[batch_indices, best_beam_idx, :]  # Shape: (batch_size, seq_len)
    
    return best_beams

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
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    predictions, references = [], []
    with torch.no_grad():
        for batch in valid_loader:
            src_input_ids = batch["src_input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            memory = encoder(input_ids=src_input_ids, attention_mask=attention_mask).last_hidden_state

            start_token_id = tokenizer.cls_token_id
            end_token_id = tokenizer.sep_token_id

            decoder_input = torch.full((src_input_ids.size(0), 1), start_token_id, device=device)
            best_beams = beam_search(decoder, memory, start_token_id, end_token_id)
            decoded_preds = tokenizer.batch_decode(best_beams, skip_special_tokens=True)
            decoded_refs = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            predictions.extend(decoded_preds)
            references.extend(decoded_refs)

    bleu_score = compute_bleu(predictions, references)
    print(f"Validation BLEU score: {bleu_score:.2f}")
