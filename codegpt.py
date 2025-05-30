from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, GenerationConfig
)
import numpy as np
from evaluate import load
import torch

# 1. Load dataset
train = load_dataset("code_x_glue_cc_code_refinement", "small", split="train[:10%]")
val = load_dataset("code_x_glue_cc_code_refinement", "small", split="validation[:10%]")
# 2. Load tokenizer and model
model_name = "microsoft/CodeGPT-small-java"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add special tokens for better task understanding
special_tokens = {
    "pad_token": "<pad>",
    "additional_special_tokens": ["<buggy>", "<fixed>", "<sep>"]
}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))
eval_inputs = []
def preprocess_eval(examples):
        # Create input with clear task instruction
    inputs = []
    targets = []
    
    for buggy, fixed in zip(examples['buggy'], examples['fixed']):
        # Format: <buggy> [buggy code] <sep> <fixed> [fixed code]
        input_text = f"<buggy> {buggy} <sep> <fixed>"
        target_text = f" {fixed}"
        eval_inputs.append(buggy)
        inputs.append(input_text)
        targets.append(target_text)
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors=None,
        padding_side='left'
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors=None,
            padding_side='left'
        )
    
    # Combine input and target tokens for causal LM training
    combined_input_ids = []
    combined_attention_mask = []
    combined_labels = []
    
    for i in range(len(inputs)):
        # Find where padding starts in input
        input_ids = model_inputs['input_ids'][i]
        attention_mask = model_inputs['attention_mask'][i]
        label_ids = labels['input_ids'][i]
        
        # Find the position of <fixed> token
        fixed_token_id = tokenizer.convert_tokens_to_ids("<fixed>")
        fixed_pos = input_ids.index(fixed_token_id) if fixed_token_id in input_ids else len(input_ids)
        
        # Combine sequences
        combined_seq = input_ids[:fixed_pos+1] + label_ids
        combined_seq = combined_seq[:256]  # Truncate to max length
        
        # Create attention mask
        combined_mask = [1] * len(combined_seq)
        
        # Create labels (mask the input part)
        combined_label = [-100] * (fixed_pos+1) + label_ids
        combined_label = combined_label[:256]
        
        # Pad sequences
        pad_length = 256 - len(combined_seq)
        combined_seq += [tokenizer.pad_token_id] * pad_length
        combined_mask += [0] * pad_length
        combined_label += [-100] * pad_length
        
        combined_input_ids.append(combined_seq[:256])
        combined_attention_mask.append(combined_mask[:256])
        combined_labels.append(combined_label[:256])
    
    return {
        'input_ids': combined_input_ids,
        'attention_mask': combined_attention_mask,
        'labels': combined_labels
    }
def preprocess_function(examples):
    # Create input with clear task instruction
    inputs = []
    targets = []
    
    for buggy, fixed in zip(examples['buggy'], examples['fixed']):
        # Format: <buggy> [buggy code] <sep> <fixed> [fixed code]
        input_text = f"<buggy> {buggy} <sep> <fixed>"
        target_text = f" {fixed}"
        inputs.append(input_text)
        targets.append(target_text)
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors=None,
        padding_side='left'
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors=None,
            padding_side='left'
        )
    
    # Combine input and target tokens for causal LM training
    combined_input_ids = []
    combined_attention_mask = []
    combined_labels = []
    
    for i in range(len(inputs)):
        # Find where padding starts in input
        input_ids = model_inputs['input_ids'][i]
        attention_mask = model_inputs['attention_mask'][i]
        label_ids = labels['input_ids'][i]
        
        # Find the position of <fixed> token
        fixed_token_id = tokenizer.convert_tokens_to_ids("<fixed>")
        fixed_pos = input_ids.index(fixed_token_id) if fixed_token_id in input_ids else len(input_ids)
        
        # Combine sequences
        combined_seq = input_ids[:fixed_pos+1] + label_ids
        combined_seq = combined_seq[:256]  # Truncate to max length
        
        # Create attention mask
        combined_mask = [1] * len(combined_seq)
        
        # Create labels (mask the input part)
        combined_label = [-100] * (fixed_pos+1) + label_ids
        combined_label = combined_label[:256]
        
        # Pad sequences
        pad_length = 256 - len(combined_seq)
        combined_seq += [tokenizer.pad_token_id] * pad_length
        combined_mask += [0] * pad_length
        combined_label += [-100] * pad_length
        
        combined_input_ids.append(combined_seq[:256])
        combined_attention_mask.append(combined_mask[:256])
        combined_labels.append(combined_label[:256])
    
    return {
        'input_ids': combined_input_ids,
        'attention_mask': combined_attention_mask,
        'labels': combined_labels
    }
class GenerativeTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # During evaluation, generate outputs
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys
            )
        
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        
        # Extract the input part (before <fixed> token)
        input_ids = inputs["input_ids"]
        batch_size = input_ids.shape[0]
        
        # Find <fixed> token position for each sample
        fixed_token_id = tokenizer.convert_tokens_to_ids("<fixed>")
        generation_inputs = []
        
        for i in range(batch_size):
            # Find where to start generation (after <fixed> token)
            fixed_pos = (input_ids[i] == fixed_token_id).nonzero(as_tuple=True)[0]
            if len(fixed_pos) > 0:
                end_pos = fixed_pos[0].item() + 1
            else:
                end_pos = len(input_ids[i])
            
            generation_inputs.append(input_ids[i][:end_pos])
        
        # Pad generation inputs
        max_len = max(len(seq) for seq in generation_inputs)
        padded_inputs = torch.full((batch_size, max_len), tokenizer.pad_token_id, device=input_ids.device)
        attention_mask = torch.zeros((batch_size, max_len), device=input_ids.device)
        
        for i, seq in enumerate(generation_inputs):
            padded_inputs[i, :len(seq)] = seq
            attention_mask[i, :len(seq)] = 1
        
        # Generate predictions
        gen_kwargs = {
            "max_length": 256,
            "num_beams": 8,
            "temperature": 0.5,
            "do_sample": False,
            "early_stopping": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=padded_inputs,
                attention_mask=attention_mask,
                **gen_kwargs
            )
        
        # Extract only the generated part
        predictions = []
        for i in range(batch_size):
            start_len = len(generation_inputs[i])
            predictions.append(generated_tokens[i][start_len:])
        
        # Pad predictions to same length
        max_pred_len = max(len(pred) for pred in predictions)
        padded_predictions = torch.full((batch_size, max_pred_len), tokenizer.pad_token_id, device=input_ids.device)
        for i, pred in enumerate(predictions):
            padded_predictions[i, :len(pred)] = pred
        
        if has_labels:
            labels = inputs["labels"]
            with torch.no_grad():
                loss = self.compute_loss(model, inputs)
                # Ensure loss is detached
                loss = loss.detach()
        else:
            loss = None
            labels = None
        
        # Ensure all tensors are detached
        if padded_predictions is not None:
            padded_predictions = padded_predictions.detach()
        if labels is not None:
            labels = labels.detach()
            
        return (loss, padded_predictions, labels)

bleumetric = load("bleu")
print(len(eval_inputs))
def compute_metrics(eval_preds):
    try:
        predictions, labels = eval_preds
        
        # Ensure we're working with numpy arrays
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # Decode predictions
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Decode labels (remove -100 values)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Clean up predictions and labels
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        with open("output.txt", "w") as file:
            # Print sample predictions for inspection
            file.write("\n" + "="*80)
            file.write("EVALUATION SAMPLES:")
            file.write("="*80)
            
            num_samples_to_print = min(100, len(decoded_preds))
            for i in range(num_samples_to_print):
                file.write(f"\n--- Sample {i+1} ---")
                
                # If we have stored inputs, print them
                if eval_inputs and i < len(eval_inputs):
                    file.write(f"ORIGINAL INPUT:\n{eval_inputs[i]}")
                    file.write("\n")
                file.write(f"TARGET (Expected output):\n{decoded_labels[i]}")
                file.write(f"\nPREDICTION (Model output):\n{decoded_preds[i]}")
                
                # Calculate sample BLEU score
                sample_bleu = bleumetric.compute(
                    predictions=[decoded_preds[i]], 
                    references=[[decoded_labels[i]]]
                )
                file.write(f"\nSample BLEU Score: {sample_bleu['bleu'] * 100:.2f}")
                file.write("-" * 40)
            
            file.write(f"\nTotal samples evaluated: {len(decoded_preds)}")
            file.write("="*80 + "\n")
            
        # Clear eval_inputs for next evaluation
        eval_inputs.clear()
        
        # Calculate overall BLEU
        bleu_score = bleumetric.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )
        
        return {"bleu": bleu_score["bleu"] * 100}
    
    except Exception as e:
        print(f"Error in compute_metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"bleu": 0.0}


# 4. Tokenize dataset
tokenizedtrain = train.map(preprocess_function, batched=True, remove_columns=train.column_names)
tokenizedval = val.map(preprocess_eval, batched=True, remove_columns=val.column_names)

# 5. Use Seq2Seq data collator
datacollator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8
)
# Training arguments with generation
trainingargs =  Seq2SeqTrainingArguments(
    output_dir="./codegpt-code-refinement-improved",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    save_steps=50000,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=40,
    warmup_steps=500,
    weight_decay=0.01,
    greater_is_better=True,
    logging_steps=300,
    fp16=True,
    predict_with_generate=True
)

# Create trainer instance
trainer = GenerativeTrainer(
    model=model,
    args=trainingargs,
    train_dataset=tokenizedtrain,
    eval_dataset=tokenizedval,
    tokenizer=tokenizer,
    data_collator=datacollator,
    compute_metrics=compute_metrics
)

# Train the model
print("Starting training...")
trainresult = trainer.train()

# Save the final model
# trainer.save_model()
# trainer.save_state()

# Evaluate on validation set
print("\nEvaluating on validation set...")
evalresults = trainer.evaluate()
print(f"Validation BLEU: {evalresults}")


# Save metrics
with open("./codegpt-code-refinement-improved/trainresults.txt", "w") as f:
    f.write(str(trainresult))
    f.write("\n\nEvaluation results:\n")
    f.write(str(evalresults))
