import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    LlamaForCausalLM, 
    LlamaConfig, 
    CodeLlamaTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq,
    AutoTokenizer
)
from datasets import load_dataset
import numpy as np
from typing import Optional, Tuple
import math
from evaluate import load

class LinformerAttention(nn.Module):
    """Linformer attention mechanism to replace standard attention"""
    
    def __init__(self, config, layeridx, k=256):
        super().__init__()
        self.config = config
        self.layeridx = layeridx
        self.hidden_size = config.hidden_size
        self.numheads = config.num_attention_heads
        self.headdim = self.hidden_size // self.numheads
        self.k = k  # Compressed sequence length
        
        # Standard attention projections
        self.qproj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.kproj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.vproj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.oproj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Linformer compression matrices
        self.E = nn.Parameter(torch.randn(config.max_position_embeddings, k))
        self.F = nn.Parameter(torch.randn(config.max_position_embeddings, k))
        
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def forward(
        self,
        hiddenstates: torch.Tensor,
        attentionmask: Optional[torch.Tensor] = None,
        positionids: Optional[torch.LongTensor] = None,
        pastkeyvalue: Optional[Tuple[torch.Tensor]] = None,
        outputattentions: bool = False,
        usecache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, qlen,  = hiddenstates.size()
        
        # Project to Q, K, V
        querystates = self.qproj(hiddenstates)
        keystates = self.kproj(hiddenstates)
        valuestates = self.vproj(hiddenstates)
        
        # Reshape for multi-head attention
        querystates = querystates.view(bsz, qlen, self.numheads, self.headdim).transpose(1, 2)
        keystates = keystates.view(bsz, qlen, self.numheads, self.headdim).transpose(1, 2)
        valuestates = valuestates.view(bsz, qlen, self.numheads, self.headdim).transpose(1, 2)
        
        # Apply Linformer compression
        if qlen > self.k:
            # Compress keys and values
            Etruncated = self.E[:qlen, :].unsqueeze(0).unsqueeze(0)  # 1, 1, q_len, k
            Ftruncated = self.F[:qlen, :].unsqueeze(0).unsqueeze(0)  # 1, 1, q_len, k
            
            # Compress: bsz, num_heads, q_len, head_dim -> bsz, num_heads, k, head_dim
            keystates = torch.matmul(Etruncated.transpose(-2, -1), keystates)
            valuestates = torch.matmul(Ftruncated.transpose(-2, -1), valuestates)
            
            # Compute attention scores with compressed K, V
            attnweights = torch.matmul(querystates, keystates.transpose(2, 3)) / math.sqrt(self.headdim)
        else:
            # Standard attention for short sequences
            attnweights = torch.matmul(querystates, keystates.transpose(2, 3)) / math.sqrt(self.headdim)
        
        # Apply causal mask
        if attentionmask is not None:
            if qlen > self.k and attentionmask.size(-1) > self.k:
                # Compress attention mask
                attentionmask = attentionmask[:, :, :, :self.k]
            attnweights = attnweights + attentionmask
        
        # Softmax
        attnweights = F.softmax(attnweights, dim=-1, dtype=torch.float32).to(querystates.dtype)
        attnweights = self.dropout(attnweights)
        
        # Apply attention to values
        attnoutput = torch.matmul(attnweights, valuestates)
        
        # Reshape and project output
        attnoutput = attnoutput.transpose(1, 2).contiguous()
        attnoutput = attnoutput.reshape(bsz, qlen, self.hidden_size)
        attnoutput = self.oproj(attnoutput)
        
        return attnoutput, attnweights if outputattentions else None, pastkeyvalue

class CodeLlamaWithLinformer(LlamaForCausalLM):
    """CodeLlama model with Linformer attention for efficient long sequence processing"""
    
    def __init__(self, config, k=256):
        super().__init__(config)
        self.k = k
        
        # Replace attention layers with Linformer
        for i, layer in enumerate(self.model.layers):
            layer.self_attn = LinformerAttention(config, i, k=k)
    
    @classmethod
    def from_pretrained(cls, modelnameorpath, k=256, **kwargs):
        """Load pretrained model and replace attention layers"""
        # Load original model first
        model = super().from_pretrained(modelnameorpath, **kwargs)
        
        # Debug version with shape checking
        for i, layer in enumerate(model.model.layers):
            original_attn = layer.self_attn
            new_attn = LinformerAttention(model.config, i, k=k)
            
            # Move to correct device and dtype
            device = original_attn.q_proj.weight.device
            dtype = original_attn.q_proj.weight.dtype
            new_attn = new_attn.to(device=device, dtype=dtype)
            
            # Debug: Print shapes and types
            print(f"Layer {i}:")
            print(f"  Original q_proj shape: {original_attn.q_proj.weight.shape}, dtype: {original_attn.q_proj.weight.dtype}")
            print(f"  New q_proj shape: {new_attn.q_proj.weight.shape}, dtype: {new_attn.q_proj.weight.dtype}")
            
            # Copy weights with shape verification
            with torch.no_grad():
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    orig_proj = getattr(original_attn, proj_name)
                    new_proj = getattr(new_attn, proj_name)
                    
                    if orig_proj.weight.shape != new_proj.weight.shape:
                        raise ValueError(f"Shape mismatch for {proj_name}: {orig_proj.weight.shape} vs {new_proj.weight.shape}")
                    
                    new_proj.weight.copy_(orig_proj.weight)
                    
                    if orig_proj.bias is not None and new_proj.bias is not None:
                        new_proj.bias.copy_(orig_proj.bias)
            
            layer.self_attn = new_attn
        
        return model

class Seq2SeqCodeLlamaTrainer:
    """Trainer for seq2seq code refinement using CodeLlama with Linformer"""
    
    def __init__(self, modelname="codellama/CodeLlama-7b-Python-hf", k=256):
        self.modelname = modelname
        self.k = k
        self.setupmodelandtokenizer()
        self.setupmetrics()
    
    def setupmodelandtokenizer(self):
        """Initialize model and tokenizer"""
        print("Loading tokenizer...")
        self.tokenizer = CodeLlamaTokenizer.from_pretrained(self.modelname)
        
        # Add special tokens for seq2seq
        specialtokens = {
            "pad_token": "<pad>",
            "bos_token": "<s>", 
            "eos_token": "</s>",
            "sep_token": "<sep>"
        }
        
        self.tokenizer.add_special_tokens(specialtokens)
        
        print(f"Loading model with Linformer (k={self.k})...")
        self.model = CodeLlamaWithLinformer.from_pretrained(
            self.modelname,
            k=self.k,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Resize embeddings for new tokens
        self.model.resizetokenembeddings(len(self.tokenizer))
    
    def setupmetrics(self):
        """Setup evaluation metrics"""
        self.bleumetric = load("bleu")
        self.rougemetric = load("rouge")
    
    def loadandpreprocessdata(self, maxlength=1024):
        """Load and preprocess the code refinement dataset"""
        print("Loading dataset...")
        dataset = load_dataset("code_x_glue_cc_code_refinement", "small")
        
        def preprocessfunction(examples):
            # Create seq2seq format: "Refine: <buggycode> <sep>" -> "<refinedcode>"
            inputs = [
                f"Refine: {buggy} <sep>"
                for buggy in examples["buggy"]
            ]
            targets = examples["fixed"]
            
            # Tokenize inputs
            modelinputs = self.tokenizer(
                inputs,
                maxlength=maxlength,
                truncation=True,
                padding=True,
                returntensors=None
            )
            
            # Tokenize targets
            labels = self.tokenizer(
                targets,
                maxlength=maxlength,
                truncation=True,
                padding=True,
                returntensors=None
            )
            
            # For seq2seq, we need to create decoder input ids and labels
            # Decoder input: <bos> + target[:-1]
            # Labels: target[1:] + <eos>
            
            decoderinputids = []
            labelslist = []
            
            for labelseq in labels["inputids"]:
                # Add BOS at the beginning for decoder input
                decoderinput = [self.tokenizer.bostokenid] + labelseq[:-1]
                # Labels are the target sequence shifted by one
                label = labelseq[1:] + [self.tokenizer.eostokenid]
                
                # Pad to same length
                maxlen = max(len(decoderinput), len(label))
                decoderinput.extend(self.tokenizer.pad_token_id,  (max_len - len(decoder_input)))
                label.extend([-100], (maxlen - len(label)))  # -100 is ignored in loss
                
                decoderinputids.append(decoderinput)
                labelslist.append(label)
            
            modelinputs["decoder_input_ids"] = decoderinputids
            modelinputs["labels"] = labelslist
            
            return modelinputs
        
        # Preprocess datasets
        self.traindataset = dataset["train"].map(
            preprocessfunction,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Preprocessing train dataset"
        )
        
        self.valdataset = dataset["validation"].map(
            preprocessfunction,
            batched=True,
            remove_columns=dataset["validation"].column_names,
            desc="Preprocessing validation dataset"
        )
        
        self.testdataset = dataset["test"].map(
            preprocessfunction,
            batched=True,
            remove_columns=dataset["test"].column_names,
            desc="Preprocessing test dataset"
        )
        
        print(f"Train samples: {len(self.traindataset)}")
        print(f"Validation samples: {len(self.valdataset)}")
        print(f"Test samples: {len(self.testdataset)}")
    
    def computemetrics(self, evalpreds):
        """Compute BLEU and ROUGE metrics"""
        predictions, labels = evalpreds
        
        # Decode predictions and labels
        decodedpreds = self.tokenizer.batchdecode(predictions, skipspecialtokens=True)
        
        # Replace -100 in labels with pad token id
        labels = np.where(labels != -100, labels, self.tokenizer.padtokenid)
        decodedlabels = self.tokenizer.batchdecode(labels, skipspecialtokens=True)
        
        # Clean up whitespace
        decodedpreds = [pred.strip() for pred in decoded_preds]
        decodedlabels = [label.strip() for label in decodedlabels]
        
        # Compute BLEU
        bleuresult = self.bleumetric.compute(
            predictions=decodedpreds,
            references=[[label] for label in decodedlabels]
        )
        
        # Compute ROUGE
        rougeresult = self.rougemetric.compute(
            predictions=decodedpreds,
            references=decodedlabels
        )
        
        return {
            "bleu": bleuresult["bleu"],
            "rouge1": rougeresult["rouge1"],
            "rouge2": rougeresult["rouge2"],
            "rougeL": rougeresult["rougeL"],
        }
    
def train(self, output_dir="./codellama-linformer-refinement", num_epochs=3):
    """Train the model"""
    
    # Data collator for seq2seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=self.tokenizer,
        model=self.model,
        padding=True,
        return_tensors="pt"
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,  # Small batch size due to long sequences
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        eval_steps=500,
        save_steps=1000,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="tensorboard"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=self.model,
        args=training_args,
        train_dataset=self.train_dataset,
        eval_dataset=self.val_dataset,
        tokenizer=self.tokenizer,
        data_collator=data_collator,
        compute_metrics=self.compute_metrics,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    print(f"Model saved to {output_dir}")
    
    return trainer
def evaluate(self, trainer=None, datasetsplit="test"):
    """Evaluate the model on test set"""
    if trainer is None:
        # Load trained model for evaluation
        self.model = CodeLlamaWithLinformer.from_pretrained(
            "./codellama-linformer-refinement",
            k=self.k
        )
    
    # Select dataset
    evaldataset = {
        "test": self.testdataset,
        "validation": self.valdataset
    }[datasetsplit]
    
    # Generate predictions
    predictions = []
    references = []
    
    print(f"Evaluating on {datasetsplit} set...")
    
    for i, example in enumerate(evaldataset):
        if i % 100 == 0:
            print(f"Processing example {i}/{len(evaldataset)}")
        
        # Prepare input
        inputtext = f"Refine: {example['buggy']} <sep>"
        inputs = self.tokenizer(
            inputtext,
            maxlength=1024,
            truncation=True,
            padding=True,
            returntensors="pt"
        ).to(self.model.device)
        
        # Generate
        with torch.nograd():
            outputs = self.model.generate(
                inputs,
                maxnewtokens=512,
                dosample=False,
                padtokenid=self.tokenizer.padtokenid,
                eostokenid=self.tokenizer.eostokenid
            )
        
        # Decode prediction
        pred = self.tokenizer.decode(outputs[0], skipspecialtokens=True)
        pred = pred.replace(inputtext, "").strip()  # Remove input part
        
        predictions.append(pred)
        references.append(example['fixed'])
    
    # Compute metrics
    bleuresult = self.bleumetric.compute(
        predictions=predictions,
        references=[ref for ref in references]
    )
    
    rougeresult = self.rougemetric.compute(
        predictions=predictions,
        references=references
    )
    
    results = {
        "bleu": bleuresult["bleu"],
        "rouge1": rougeresult["rouge1"],
        "rouge2": rougeresult["rouge2"],
        "rougeL": rougeresult["rougeL"],
    }
    
    print(f"Evaluation Results on {datasetsplit}:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    
    return results

def computemetrics(self, evalpreds):
    """Compute BLEU and ROUGE metrics during training"""
    predictions, labels = evalpreds
    
    # Decode predictions and labels
    decodedpreds = self.tokenizer.batchdecode(predictions, skipspecialtokens=True)
    
    # Replace -100 in labels with pad token id
    labels = np.where(labels != -100, labels, self.tokenizer.padtokenid)
    decodedlabels = self.tokenizer.batchdecode(labels, skipspecialtokens=True)
    
    # Clean up whitespace
    decodedpreds = [pred.strip() for pred in decodedpreds]
    decodedlabels = [label.strip() for label in decodedlabels]
    
    # Compute BLEU
    bleuresult = self.bleumetric.compute(
        predictions=decodedpreds,
        references=[[label] for label in decodedlabels]
    )
    
    # Compute ROUGE
    rougeresult = self.rougemetric.compute(
        predictions=decodedpreds,
        references=decodedlabels
    )
    
    return {
        "bleu": bleuresult["bleu"],
        "rouge1": rougeresult["rouge1"],
        "rouge2": rougeresult["rouge2"],
        "rougeL": rougeresult["rougeL"],
    }
# Training loop usage
if __name__ == "__main__":
    # Initialize trainer
    trainer_instance = Seq2SeqCodeLlamaTrainer()
    
    # Load and preprocess data
    trainer_instance.loadandpreprocessdata()
    
    # Train the model
    trainer = trainer_instance.train(
        output_dir="./codellama-linformer-refinement",
        num_epochs=3
    )
    
    # Evaluate on test set
    test_results = trainer_instance.evaluate(trainer, "test")
    
    # Evaluate on validation set
    val_results = trainer_instance.evaluate(trainer, "validation")
    
    print("\nFinal Results:")
    print("Test Results:", test_results)
    print("Validation Results:", val_results)
