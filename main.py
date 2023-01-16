import argparse
import preprocess, eval_bleu
from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
import evaluate
import numpy as np

from datasets import load_dataset, concatenate_datasets # Huggingface datasets
import transformers
from transformers import MBart50Tokenizer, MBartConfig, MBartForConditionalGeneration, DataCollatorForSeq2Seq

from functools import partial


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--source-file", required=True, help="file to be translated")
    #parser.add_argument("--docids-file", required=True, help="file with document ids")
    #parser.add_argument("--predictions-file", required=True, help="file to save the predictions")
    #parser.add_argument("--reference-file", default=None, help="reference file, used if with --gold-target-context")
    parser.add_argument("-s", "--source_lang", required=True, help="souce language")
    parser.add_argument("-t", "--target_lang", required=True)
    parser.add_argument("-p", "--path", required=True, metavar="FILE", help="path to model file for bsd is '/home/sumire/discourse_context_mt/data/BSD-master/'")
    #parser.add_argument("-c", "--context_size", default=0, help="the number of the context sentence for each input")
    parser.add_argument("-c", "--context_size", type=int, default=0, help="the number of the context sentence for each input")
    args = parser.parse_args()

    source_lang = args.source_lang
    target_lang = args.target_lang
    file_path = args.path
    context_size = args.context_size  
    
    
    # Load the dataset
    file_path = file_path
    data_files = {"train": f"{file_path}train.json", "validation": f"{file_path}dev.json", "test": f"{file_path}test.json"}
    dataset = load_dataset("json", data_files=data_files)
    
    # Tokenize using "facebook/mbart-large-50-many-to-many-mmt"
    model_checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
    configuration = MBartConfig()
    tokenizer = MBart50Tokenizer.from_pretrained(model_checkpoint, src_lang=f"{source_lang}_XX", tgt_lang=f"{target_lang}_XX")
    model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)
    
    
    
    # Check if preprocessing correctly
    #inputs=preprocess.preprocess_function(dataset["validation"], context_size, tokenizer)
    #print (inputs)
    #print (model_inputs["input_ids"][1])
    #print (tokenizer.decode(model_inputs['input_ids'][1]))

    #print (tokenizer.decode(model_inputs['labels'][1]))
    #print (tokenizer.decode(model_inputs['input_ids'][1]))


    # Apply the preprocess function for the entire dataset 
    tokenized_datasets = dataset.map(
    partial(preprocess.preprocess_function, context_size, tokenizer),
    batched=True,
    remove_columns=dataset["train"].column_names,
    )
    

    #print (tokenizer.decode(tokenized_datasets["train"][:5]['input_ids'][3]))

    # Create a batch using DataCollator and pad dinamically
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt") 
    
    # New Trainer
    training_args = Seq2SeqTrainingArguments(
    output_dir='./results/inputs_context_1',
    evaluation_strategy="steps",
    learning_rate=2e-5,        
    logging_dir='./logs',                       
    per_device_train_batch_size=4, #16  
    per_device_eval_batch_size=4,  #64 
    warmup_steps=500,                
    weight_decay=0.01,
    save_total_limit=3,
    report_to="all",
    fp16=True,
    do_eval=True,
    metric_for_best_model = 'bleu', # eval_bleu.metric,
    load_best_model_at_end = True,
    num_train_epochs = 10,
    greater_is_better = True,
    predict_with_generate = True,
    do_train = True,
    logging_strategy = 'steps',
    save_strategy = 'steps',
    logging_steps=1000,
    #gradient_accumulation_steps=1000,
    #half_precision_backend="apex",
    eval_steps=1000,
    save_steps=1000,
    #report_to=“tensorboard” # Add this
    )

    trainer = Seq2SeqTrainer(
        model=model,                         
        args=training_args,                  
        train_dataset=tokenized_datasets["train"],        
        eval_dataset=tokenized_datasets["validation"],            
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=partial(eval_bleu.compute_metrics, tokenizer),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)] # Put tensorboard logger: [EarlyStoppingCallback(early_stopping_patience=10) , CustomLoggerCallback]
        )
    
    trainer.train()
    
    
if __name__ == "__main__":
    main()
