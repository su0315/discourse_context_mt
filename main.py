import argparse
import preprocess, eval_bleu
from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback, MBart50Tokenizer, MBartConfig, MBartForConditionalGeneration, DataCollatorForSeq2Seq, pipeline
import evaluate
import numpy as np
import json
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
    source_lang = cfg.generic.src_lang
    target_lang = cfg.generic.tgt_lang
    file_path = cfg.generic.dataset
    src_context_size = cfg.generic.src_context
    tgt_context_size = cfg.generic.tgt_context
    cw_dropout_rate = cfg.generic.dropout

   
    # Tokenize using "facebook/mbart-large-50-many-to-many-mmt"
    model_checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
    configuration = MBartConfig()
    tokenizer = MBart50Tokenizer.from_pretrained(model_checkpoint, src_lang=f"{source_lang}_XX", tgt_lang=f"{target_lang}_XX", return_tensors="pt")
    model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)
    
    # Add special token to separate context and current sentence
    tokenizer.add_special_tokens({"sep_token":"</t>"})
    model.resize_token_embeddings(len(tokenizer))
    print("sep_token",tokenizer.get_added_vocab(), tokenizer.convert_tokens_to_ids("</t>"),tokenizer.decode(tokenizer.sep_token_id))

    
    # Load the dataset
    file_path = file_path
    data_files = {"train": f"{file_path}train.json", "validation": f"{file_path}dev.json", "test": f"{file_path}test.json"} # modify train_short -> train.json
    dataset = load_dataset("json", data_files=data_files)


    # Apply the preprocess function for the entire dataset 
    tokenized_datasets = dataset.map(
    partial(preprocess.preprocess_function, src_context_size, tgt_context_size, 0, tokenizer),
    batched=True,
    remove_columns=dataset["train"].column_names, # train
    )
    
    print ("First preprocess done!")
    

    # CoWord Dropout for train data
    if cw_dropout_rate > 0:
        tokenized_datasets['train'] = dataset['train'].map(
        partial(preprocess.preprocess_function, src_context_size, tgt_context_size, cw_dropout_rate, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names, # train
        )
        print ("CoWord done!")

    # Check if the decoded input has CoWord dropout
    print ("masked_decoded_inputs:", tokenizer.batch_decode(tokenized_datasets["train"]["input_ids"][:5]))

    # Create a batch using DataCollator and pad dinamically
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt") 
    
    training_args, callbacks = initialize_trainer(cfg)

    trainer = Seq2SeqTrainer(
        model=model,                         
        args=training_args,                  
        train_dataset=tokenized_datasets["train"],        
        eval_dataset=tokenized_datasets["validation"],            
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=partial(eval_bleu.compute_metrics, tokenizer),
        callbacks = callbacks # Put tensorboard logger: [EarlyStoppingCallback(early_stopping_patience=10) , CustomLoggerCallback]
        #callbacks = integrations.TensorBoardCallback # for tensorboard call backs, don't know how to run this

        )

    #compute the metrics for the translations you got 
    
    """
    #process the test file --> use tokenized_datasets["test"]
    model.eval()
    print (model)
    #use the model to get translations on tokenized_datasets["test"] --> save
    translator = pipeline("translation", model=model, tokenizer = MBart50Tokenizer.from_pretrained(model, src_lang=f"{source_lang}_XX", tgt_lang=f"{target_lang}_XX"))
    test_inputs = [sent['en_sentence'] for doc in dataset["test"]["conversation"] for sent in doc]
    translations = translator(test_inputs)
    """
    
    # Inference on Test set
    model.eval()
    test_inputs = tokenized_datasets["test"]["input_ids"]
    device = torch.device("cuda")
    model.cuda()
    """
    print (torch.LongTensor(test_inputs).size())
    generated_tokens = model.generate(torch.LongTensor(test_inputs).unsqueeze(0).cuda())
    print (torch.LongTensor(test_inputs).size())
    translations = tokenizer.batch_decode(generated_tokens, max_length=128 ,forced_bos_token_id=tokenizer.lang_code_to_id[f"{target_lang}_XX"])
    """
    translations = []
    for input in test_inputs:
        generated_tokens = model.generate(torch.LongTensor(input).unsqueeze(0).cuda(), max_length=128 ,forced_bos_token_id=tokenizer.lang_code_to_id[f"{target_lang}_XX"])#
        translations.append(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
    
    print ("before translation", tokenizer.batch_decode(tokenized_datasets["test"]["input_ids"][:5]))
    print ("translated:", translations[:5])
    
    # Decode inference translations 
    with open(cfg.training_args.output_dir+'/translations.txt','w', encoding='utf8') as wf:
         for translation in translations:
            for item in translation:
                wf.write(item.strip()+'\n') #ensure_ascii=False
                print(item)
    #compute the metrics for the translations you got 

    
    
if __name__ == "__main__":
    main()





