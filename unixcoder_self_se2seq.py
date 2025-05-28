import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from unixcoder import UniXcoder

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CodeRefinementDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        buggy_code = str(row['buggy'])
        fixed_code = str(row['fixed'])
        
        # Tokenize input (buggy code)
        tokens_ids = self.tokenizer.tokenize([buggy_code], max_length=self.max_length, mode="<encoder-decoder>")[0]
        source_ids = torch.tensor(tokens_ids).long()
        
        # Tokenize target (fixed code)
        target_tokens = self.tokenizer.tokenize([fixed_code], max_length=self.max_length, mode="<encoder-decoder>")[0]
        target_ids = torch.tensor(target_tokens).long()
        
        return {
            'source_ids': source_ids,
            'target_ids': target_ids
        }
class UniXcoderSeq2Seq(nn.Module):
    def init(self, modelpath='microsoft/unixcoder-base'):
        super(UniXcoderSeq2Seq, self).init()
        
        # Initialize UniXcoder model
        self.model = UniXcoder(modelpath)
        self.config = self.model.config
        
        # Create decoder layer with cross-attention
        self.lmhead = nn.Linear(self.config.hiddensize, self.config.vocabsize, bias=False)
        self.lmhead.weight = self.model.embeddings.wordembeddings.weight
        
    def forward(self, sourceids, targetids=None):
        # Get attention masks
        sourcemask = sourceids.ne(self.config.padtokenid)
        
        if targetids is not None:
            # Training mode - use teacher forcing
            targetmask = targetids.ne(self.config.padtokenid)
            
            # Encode source
            outputs = self.model(sourceids, attentionmask=sourcemask)
            encoderoutput = outputs0  # (batchsize, sourcelength, hiddensize)
            
            # Decode with cross-attention
            # Shift targetids right for input
            decoderinputids = self.shifttokensright(targetids)
            decodermask = decoderinputids.ne(self.config.padtokenid)
            
            # Get decoder outputs
            decoderoutputs = self.model(
                decoderinputids,
                attentionmask=decodermask,
                encoderhiddenstates=encoderoutput,
                encoderattentionmask=sourcemask
            )
            
            sequenceoutput = decoderoutputs0
            predictionscores = self.lmhead(sequenceoutput)
            
            # Calculate loss
            lossfct = nn.CrossEntropyLoss(ignoreindex=self.config.padtokenid)
            loss = lossfct(predictionscores.view(-1, self.config.vocabsize), 
                          targetids.view(-1))
            
            return loss, predictionscores
        else:
            # Inference mode
            outputs = self.model(sourceids, attentionmask=sourcemask)
            encoderoutput = outputs0
            return encoderoutput
    
    def shifttokensright(self, inputids):
        """Shift input ids right for decoder input"""
        shiftedinputids = inputids.newzeros(inputids.shape)
        shiftedinputids:, 1: = inputids[:, :-1].clone()
        shiftedinputids[:, 0] = self.config.bostokenid
        return shiftedinputids
    
    def generate(self, sourceids, maxlength=512, numbeams=5):
        """Generate code using beam search"""
        sourcemask = sourceids.ne(self.config.padtokenid)
        
        # Encode source
        outputs = self.model(sourceids, attentionmask=sourcemask)
        encoderoutput = outputs0
        
        # Initialize decoder input
        batchsize = sourceids.size(0)
        decoderinputids = torch.full((batchsize, 1), self.config.bostokenid, 
                                     dtype=torch.long, device=sourceids.device)
        
        # Simple greedy decoding (you can implement beam search if needed)
        for  in range(maxlength - 1):
            decodermask = decoderinputids.ne(self.config.padtokenid)
            
            decoderoutputs = self.model(
                decoderinputids,
                attentionmask=decodermask,
                encoderhiddenstates=encoderoutput,
                encoderattentionmask=sourcemask
            )
            predictions = self.lm_head(decoder_outputs[0])
            next_token = predictions[:, -1, :].argmax(dim=-1, keepdim=True)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
            
            # Stop if all sequences have generated EOS token
            if (next_token == self.config.eos_token_id).all():
                break
        
            return decoder_input_ids
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