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

DATA_FILE             = "./chat_data.csv"               # columns: prompt (patient says),response (doctor says)
BASE_MODEL            = "microsoft/DialoGPT-small"  
OUTPUT_DIR            = "./medbot_finetuned"            
NUM_EPOCHS            = 3
BATCH_SIZE            = 4
LEARNING_RATE         = 5e-5
MAX_LENGTH            = 512                             
GRADIENT_ACCUM_STEPS  = 8                          
LOGGING_STEPS         = 50
SAVE_STEPS            = 500
SAVE_TOTAL_LIMIT      = 2

df = pd.read_csv(DATA_FILE)
dataset = Dataset.from_pandas(df)  
train_data = dataset

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
model     = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

def tokenize_fn(example):
    text = example["prompt"] + tokenizer.eos_token + example["response"] + tokenizer.eos_token
    return tokenizer(text, truncation=True, max_length=MAX_LENGTH)

tokenized_data = train_data.map(
    tokenize_fn,
    batched=False,
    remove_columns=train_data.column_names
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

use_fp16 = torch.cuda.is_available()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=use_fp16,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    dataloader_num_workers=4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    data_collator=data_collator
)


trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# TEST INFERENCE CODE

tok = AutoTokenizer.from_pretrained("./medbot_finetuned")
mdl = AutoModelForCausalLM.from_pretrained("./medbot_finetuned")

input_ids = tok("Patient: I feel dizzy.",return_tensors='pt')["input_ids"]
reply_ids = mdl.generate(input_ids, max_length=50)
print(tok.decode(reply_ids[0], skip_special_tokens=True))
