import preprocess, eval_bleu

from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import numpy as np

training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,        
    #logging_dir='./logs',            
    num_train_epochs=3, #3             
    per_device_train_batch_size=4, #16  
    per_device_eval_batch_size=4,  #64 
    warmup_steps=500,                
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=True,
    report_to="all",
    fp16=True,
    #gradient_accumulation_steps=1000,
    #half_precision_backend="apex"
)

trainer = Seq2SeqTrainer(
    model=preprocess.model,                         
    args=training_args,                  
    train_dataset=preprocess.tokenized_datasets["train"],        
    eval_dataset=preprocess.tokenized_datasets["validation"],            
    data_collator=preprocess.data_collator,
    tokenizer=preprocess.tokenizer,
    compute_metrics=eval_bleu.compute_metrics
)

trainer.train()