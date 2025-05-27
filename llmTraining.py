import os
import multiprocessing as mp
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# Fixing some multiprocessing errors with Huggingface by disabling it
mp.set_start_method("spawn", force=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_FILE             = "./chat_data.csv"               # columns: prompt (patient says),response (doctor says)
BASE_MODEL            = "TBD"  
OUTPUT_DIR            = "./medbot_finetuned"            
NUM_EPOCHS            = 3
BATCH_SIZE            = 4
LEARNING_RATE         = 5e-5
MAX_LENGTH            = 512                             
GRADIENT_ACCUM_STEPS  = 8                          
LOGGING_STEPS         = 50
SAVE_STEPS            = 500
SAVE_TOTAL_LIMIT      = 2
USE_FP16              = torch.cuda.is_available()

df = pd.read_csv(DATA_FILE)
dataset = Dataset.from_pandas(df)  
train_data = dataset

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
model     = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

def tokenize_fn(ex):
    txt = ex["prompt"] + tokenizer.eos_token + ex["response"] + tokenizer.eos_token
    
    tokens = tokenizer(
        txt, 
        truncation=True, 
        max_length=MAX_LENGTH, 
        padding="max_length", 
        return_tensors=None  
    )
    
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    
    labels = input_ids.copy()
    
    prompt_with_eos = ex["prompt"] + tokenizer.eos_token
    prompt_tokens = tokenizer(prompt_with_eos, add_special_tokens=False)
    prompt_length = len(prompt_tokens["input_ids"])
    
    for i in range(min(prompt_length, len(labels))):
        labels[i] = -100
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
tokenized_data = train_data.map(
    tokenize_fn,
    batched=False,
    remove_columns=train_data.column_names
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        report_to=["none"],
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=USE_FP16,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        dataloader_num_workers=0,
        warmup_steps=100,  
        weight_decay=0.01,  
        eval_strategy="no",  
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        processing_class=tokenizer  
    )


trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# TEST INFERENCE CODE

tok = AutoTokenizer.from_pretrained("./medbot_finetuned")
mdl = AutoModelForCausalLM.from_pretrained("./medbot_finetuned")

inp = tok(
            "I feel dizzy." + tok.eos_token,  
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )

with torch.no_grad():
    reply_ids = mdl.generate(
        inp["input_ids"],
        attention_mask=inp["attention_mask"],
        max_new_tokens=50, 
        min_length=inp["input_ids"].shape[1] + 5,  
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        do_sample=True, 
        temperature=0.7,  
        top_p=0.9,  
        repetition_penalty=1.1,  
        no_repeat_ngram_size=2  
    )
response_ids = reply_ids[0][inp["input_ids"].shape[1]:]
response = tok.decode(response_ids, skip_special_tokens=True)
print(response)
