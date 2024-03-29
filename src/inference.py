# Not using Custom Trainer, but using original Trainer
import argparse
import preprocess, eval_bleu
from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback, MBart50Tokenizer, MBartConfig, MBartForConditionalGeneration, DataCollatorForSeq2Seq
import evaluate
import numpy as np
import json
#from custom_model2 import MBartModelC, MBartForConditionalGenerationC
#from custom_trainer2 import Seq2SeqTrainerC
#from logger import CustomLoggerCallback
from transformers import integrations
from datasets import load_dataset, concatenate_datasets 
from functools import partial
from jsonargparse import (ActionConfigFile, ArgumentParser, Namespace,namespace_to_dict)
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
    parser.add_argument("--generic.tag", type=bool, default=False)
    
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
    tag = cfg.generic.tag
    

   
    # Test the pretrained model
    model_checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
    #model_checkpoint = "/mnt/data-poseidon/sumire/bsd_en-ja/newest_truncate_padding_mex_length/src_attention/5-1-t/checkpoint-10000"
    configuration = MBartConfig()
    tokenizer = MBart50Tokenizer.from_pretrained(model_checkpoint, src_lang=f"{src_lang}_XX", tgt_lang=f"{tgt_lang}_XX")
    model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)
    
    if model_checkpoint != "facebook/mbart-large-50-many-to-many-mmt":
        print ("Don't add special additional tokens")
        # Add special token to separate context and current sentence
        tokenizer.add_special_tokens({"sep_token":"</t>"})
        #tokenizer.add_special_tokens({"sep1_token":"</t1>"})

        # Add contextual special tokens
        special_tokens = []

        # Add Speaker Tags
        speaker_tags = ['<CurrSpeak>','<DiffSpeak>']
        for i in speaker_tags:
            special_tokens.append(i)
        #special_tokens_dict = {'additional_special_tokens': ['<CurrSpeak>','<DiffSpeak>']}
        #tokenizer.add_special_tokens(special_tokens_dict)

        # Add Scene Tags
        scene_tags = ['<face-to-face conversation>','<phone call>', '<general chatting>', '<meeting>', '<training>', '<presentation>']
        for i in scene_tags:
            special_tokens.append(i)
        #special_tokens_dict = {'additional_special_tokens': ['<face-to-face conversation>','<phone call>', '<general chatting>', '<meeting>', '<training>', '<presentation>']}
        #tokenizer.add_special_tokens(special_tokens_dict)
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        tokenizer.add_special_tokens(special_tokens_dict)

        #print ("speacial_tokens_dict", special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
    

    # Load the dataset for training
    file_path = file_path
    data_files = {"test": f"{file_path}test.json"}
    dataset = load_dataset("json", data_files=data_files)


    # Apply the preprocess function for the entire dataset 
    tokenized_datasets = dataset.map(
    partial(preprocess.preprocess_function, src_lang, tgt_lang, tag, speaker, src_context_size, tgt_context_size, random_context, 0.0, tgt_sep, tokenizer),
    batched=True,
    remove_columns=dataset["test"].column_names, # train
    )
    
    # Create a batch using DataCollator and pad dinamically
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt") 
    
    training_args, callbacks = initialize_trainer(cfg)

    trainer = Seq2SeqTrainer(
        model=model,                         
        args=training_args,                              
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=partial(eval_bleu.compute_metrics, output_dir, tgt_lang, tokenizer),
        callbacks = callbacks, # Put tensorboard logger: [EarlyStoppingCallback(early_stopping_patience=10) , CustomLoggerCallback]
        #callbacks = integrations.TensorBoardCallback # for tensorboard call backs, don't know how to run this

        )
    

    #model.eval()
    preds, label_ids, metrics = trainer.predict(tokenized_datasets["test"])
    print ("predict")
    print ("preds:", preds)
    print ("label_ids:", label_ids)
    print ("metrics:", metrics)

if __name__ == "__main__":
    main()

