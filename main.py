import argparse
import preprocess, eval_bleu
from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback, MBart50Tokenizer, MBartConfig, MBartForConditionalGeneration, DataCollatorForSeq2Seq
import evaluate
import numpy as np
import json
from custom_model import MBartModelC, MBartForConditionalGenerationC
from custom_trainer import Seq2SeqTrainerC
#from logger import CustomLoggerCallback
from transformers import integrations
from datasets import load_dataset, concatenate_datasets 
from functools import partial
from jsonargparse import (ActionConfigFile, ArgumentParser, Namespace,
                          namespace_to_dict)
from datasets import disable_caching
disable_caching()
import torch

def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Command for training models.")
    parser.add_class_arguments(Seq2SeqTrainingArguments, "training_args")
    parser.add_class_arguments(EarlyStoppingCallback, "early_stopping")
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--generic.tgt_lang", required=True, help="souce language")
    parser.add_argument("--generic.src_lang", required=True)
    parser.add_argument("--generic.dataset", required=True, metavar="FILE", help="path to model file for bsd is '/home/sumire/discourse_context_mt/data/BSD-master/'")
    parser.add_argument("--generic.src_context",type=int, default="src", help="the number of the target context sentence for each input")
    parser.add_argument("--generic.tgt_context", type=int, default=0, help="the number of the source context sentence for each input")
    parser.add_argument("--generic.dropout", type=float, choices=np.arange(0.0, 1.0, 0.1), default=0, help="the coword dropout rate")
    parser.add_argument("--generic.tgt_sep", type=bool, default=False)# SU: changed default = True since error: pyarrow.lib.ArrowInvalid: Column 3 named context_ids expected length 70 but got length 1
    parser.add_argument("--generic.speaker", type=bool, default=False)
    parser.add_argument("--generic.random_context", type=bool, default=False)

    #parser.add_argument("--training_args", type=Seq2SeqTrainingArguments)
    
    return parser
    
    

def initialize_trainer(configs) -> TrainingArguments:
    
    trainer_args = Seq2SeqTrainingArguments(**namespace_to_dict(configs.training_args))
    early_stop_callback = EarlyStoppingCallback(
        **namespace_to_dict(configs.early_stopping)
    )
    callbacks = [early_stop_callback]

    return trainer_args, callbacks

def main():
    parser = read_arguments()
     
    cfg = parser.parse_args()

    print(cfg)
    src_lang = cfg.generic.src_lang
    tgt_lang = cfg.generic.tgt_lang
    file_path = cfg.generic.dataset
    src_context_size = cfg.generic.src_context
    tgt_context_size = cfg.generic.tgt_context
    cw_dropout_rate = cfg.generic.dropout
    tgt_sep = cfg.generic.tgt_sep
    speaker = cfg.generic.speaker
    random_context = cfg.generic.random_context
    output_dir = cfg.training_args.output_dir
    

   
    # Tokenize using "facebook/mbart-large-50-many-to-many-mmt"
    model_checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
    configuration = MBartConfig()
    tokenizer = MBart50Tokenizer.from_pretrained(model_checkpoint, src_lang=f"{src_lang}_XX", tgt_lang=f"{tgt_lang}_XX")
    model = MBartForConditionalGenerationC.from_pretrained(model_checkpoint)
    
    # Add special token to separate context and current sentence
    tokenizer.add_special_tokens({"sep_token":"</t>"})
    #tokenizer.add_special_tokens({"sep1_token":"</t1>"})

    # Add special token for speaker 1 to 5
    """
    if speaker:
        max_num_speaker = 5
        for i in range(max_num_speaker):
            tokenizer.add_special_tokens({f"speaker{i}_token":f"<S{i}>"})
            #print(f"speaker_token{i}",tokenizer.get_added_vocab(), tokenizer.convert_tokens_to_ids(f"<S{i}>"), tokenizer.decode(tokenizer.speaker1_token_id))
        
        #tokenizer.add_special_tokens({"speaker0_token":"<S0>"})
    """
    model.resize_token_embeddings(len(tokenizer))
    

    # Load the train and eval dataset for training
    file_path = file_path
    data_files = {"train": f"{file_path}short_train.json", "validation": f"{file_path}short_dev.json", "test": f"{file_path}short_test.json"}
    dataset = load_dataset("json", data_files=data_files)


    # Apply the preprocess function for the entire dataset 
    tokenized_datasets = dataset.map(
    partial(preprocess.preprocess_function, src_lang, tgt_lang, speaker, src_context_size, tgt_context_size, random_context, 0.0, tgt_sep, tokenizer),
    batched=True,
    remove_columns=dataset["train"].column_names, # train
    )
    
    tokenized_datasets['train'] = dataset['train'].map(
    partial(preprocess.preprocess_function, src_lang, tgt_lang, speaker, src_context_size, tgt_context_size, random_context, cw_dropout_rate, tgt_sep, tokenizer),
    batched=True,
    remove_columns=dataset["train"].column_names, # train
    )
    print(tokenized_datasets.keys())

    # Create a batch using DataCollator and pad dinamically
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt") 
    
    training_args, callbacks = initialize_trainer(cfg)

    trainer = Seq2SeqTrainerC(
        model=model,                         
        args=training_args,                  
        train_dataset=tokenized_datasets["train"],        
        eval_dataset=tokenized_datasets["validation"],            
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=partial(eval_bleu.compute_metrics, output_dir, tgt_lang, tokenizer),
        callbacks = callbacks, # Put tensorboard logger: [EarlyStoppingCallback(early_stopping_patience=10) , CustomLoggerCallback]
        #callbacks = integrations.TensorBoardCallback # for tensorboard call backs, don't know how to run this

        )
    
    trainer.train()
    model.eval()
    preds, labels, scores = trainer.predict(tokenized_datasets["test"])


    """
    # Inference on Test dataset 
    # Load Test data
    file_path = file_path
    test_data_files = {"test": f"{file_path}test.json"}
    test_data = load_dataset("json", data_files=test_data_files)

    print ("test_data", test_data["test"])
    inference.eval_on_inference(
        model=model, 
        tokenizer=tokenizer, 
        data=test_data["test"], 
        src_lang=src_lang, 
        tgt_lang=tgt_lang, 
        output_dir=output_dir)
    """

if __name__ == "__main__":
    main()

