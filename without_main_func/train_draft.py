import preprocess_draft, eval_bleu_draft

from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
import evaluate
import numpy as np


# Previous trainer
"""
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
    model=preprocess_draft.model,                         
    args=training_args,                  
    train_dataset=preprocess_draft.tokenized_datasets["train"],        
    eval_dataset=preprocess_draft.tokenized_datasets["validation"],            
    data_collator=preprocess_draft.data_collator,
    tokenizer=preprocess_draft.tokenizer,
    compute_metrics=eval_bleu_draft.compute_metrics
)
"""
# New Trainer
training_args = Seq2SeqTrainingArguments(
    output_dir='./results/inputs_context_1',
    evaluation_strategy="steps",
    learning_rate=2e-5,        
    logging_dir='./logs',                       
    per_device_train_batch_size=4, #16  
    per_device_eval_batch_size=4,  #64 
    warmup_steps=500,
    eval_steps = 1000,
    save_steps = 1000,
    weight_decay=0.01,
    save_total_limit=3,
    report_to="all",
    fp16=True,
    do_eval=True,
    metric_for_best_model = 'bleu', # eval_bleu_draft.metric,
    load_best_model_at_end = True,
    num_train_epochs = 30,
    greater_is_better = True,
    predict_with_generate = True,
    do_train = True,
    logging_strategy = 'steps',
    save_strategy = 'steps',
    logging_steps=1000,
    #gradient_accumulation_steps=1000,
    #half_precision_backend="apex",
    #report_to=“tensorboard” # Add this
)

trainer = Seq2SeqTrainer(
    model=preprocess_draft.model,                         
    args=training_args,                  
    train_dataset=preprocess_draft.tokenized_datasets["train"],        
    eval_dataset=preprocess_draft.tokenized_datasets["validation"],            
    data_collator=preprocess_draft.data_collator,
    tokenizer=preprocess_draft.tokenizer,
    compute_metrics=eval_bleu_draft.compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]
)


def main():
    trainer.train()

if __name__ == "__main__":
    main()