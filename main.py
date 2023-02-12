import argparse
import preprocess, eval_bleu
from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback, MBart50Tokenizer, MBartConfig, MBartForConditionalGeneration, DataCollatorForSeq2Seq
import evaluate
import numpy as np
from logger import CustomLoggerCallback
from transformers import integrations
from datasets import load_dataset, concatenate_datasets 
from functools import partial


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--docids-file", required=True, help="file with document ids")
    #parser.add_argument("--predictions-file", required=True, help="file to save the predictions")
    parser.add_argument("-s", "--source_lang", required=True, help="souce language")
    parser.add_argument("-t", "--target_lang", required=True)
    parser.add_argument("-p", "--path", required=True, metavar="FILE", help="path to model file for bsd is '/home/sumire/discourse_context_mt/data/BSD-master/'")
    parser.add_argument("-sc", "--src_context_size",type=int, default="src", help="the number of the target context sentence for each input")
    parser.add_argument("-tc", "--tgt_context_size", type=int, default=0, help="the number of the source context sentence for each input")
    parser.add_argument("-d", "--cw_dropout_rate", type=float, choices=np.arange(0.0, 1.0, 0.1), default=0, help="the coword dropout rate")
    args = parser.parse_args()

    source_lang = args.source_lang
    target_lang = args.target_lang
    file_path = args.path
    src_context_size = args.src_context_size  
    tgt_context_size = args.tgt_context_size
    cw_dropout_rate = args.cw_dropout_rate

    # Load the dataset
    file_path = file_path
    data_files = {"train": f"{file_path}train.json", "validation": f"{file_path}dev.json", "test": f"{file_path}test.json"}
    dataset = load_dataset("json", data_files=data_files)

    # Tokenize using "facebook/mbart-large-50-many-to-many-mmt"
    model_checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
    configuration = MBartConfig()
    tokenizer = MBart50Tokenizer.from_pretrained(model_checkpoint, src_lang=f"{source_lang}_XX", tgt_lang=f"{target_lang}_XX")
    model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)
    
    # Add special token to separate context and current sentence
    tokenizer.add_special_tokens({"sep_token":"</t>"})
    model.resize_token_embeddings(len(tokenizer))
    print("sep_token",tokenizer.get_added_vocab(), tokenizer.convert_tokens_to_ids("</t>"),tokenizer.decode(tokenizer.sep_token_id))

    # Apply the preprocess function for the entire dataset 
    tokenized_datasets = dataset.map(
    partial(preprocess.preprocess_function, src_context_size, tgt_context_size, cw_dropout_rate, tokenizer),
    batched=True,
    remove_columns=dataset["train"].column_names, # train
    )

    """
    # Coword Dropout
    if cw_dropout_rate > 0:
        coword_dataset = tokenized_dataset[train].map(partial(preprocess.coword_dropout(), cw_drop_rate)) 
        tokenized_dataset["train"] = coword_dataset
    """

    # Check the decoded input
    #print (tokenizer.decode(tokenized_datasets["train"][:5]['input_ids'][3]))

    # Create a batch using DataCollator and pad dinamically
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt") 
    
    # New Trainer
    training_args = Seq2SeqTrainingArguments(
    output_dir='./results/bsd_en-ja/2-1_dropout', # Modify here
    evaluation_strategy="steps",
    learning_rate=2e-5,        
    logging_dir='./logs',                       
    per_device_train_batch_size=4, #16  
    per_device_eval_batch_size=4,  #64 
    warmup_steps=500,                
    weight_decay=0.01,
    save_total_limit=3,
    report_to="all", # occasionally "tensorboard"
    fp16=True,
    do_eval=True,
    #do_predict = True,
    metric_for_best_model = 'comet', # eval_bleu.metric,
    load_best_model_at_end = True,
    num_train_epochs = 10,
    greater_is_better = True,
    predict_with_generate = True,
    do_train = True, # True
    logging_strategy = 'steps',
    save_strategy = 'steps',
    logging_steps=1000,
    #gradient_accumulation_steps=1000,
    #half_precision_backend="apex",
    eval_steps=1000,
    save_steps=1000, 
    include_inputs_for_metrics = True # for comet
    )

    trainer = Seq2SeqTrainer(
        model=model,                         
        args=training_args,                  
        train_dataset=tokenized_datasets["train"],        
        eval_dataset=tokenized_datasets["validation"],          
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=partial(eval_bleu.compute_metrics, tokenizer),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)], # Put tensorboard logger: [EarlyStoppingCallback(early_stopping_patience=10) , CustomLoggerCallback]
        #callbacks = integrations.TensorBoardCallback # for tensorboard call backs, don't know how to run this

        )
    
    trainer.train()
    
    
if __name__ == "__main__":
    main()
