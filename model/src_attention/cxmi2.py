import argparse
import preprocess_2, eval_bleu
from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback, MBart50Tokenizer, MBartConfig, MBartForConditionalGeneration, DataCollatorForSeq2Seq
import evaluate
import numpy as np
import json
from custom_model2 import MBartModelC, MBartForConditionalGenerationC
from custom_trainer2 import Seq2SeqTrainerC
#from logger import CustomLoggerCallback
from transformers import integrations
from datasets import load_dataset, concatenate_datasets 
from functools import partial
from jsonargparse import (ActionConfigFile, ArgumentParser, Namespace,
                          namespace_to_dict)
from datasets import disable_caching, Dataset
disable_caching()
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Command for training models.")
    
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_class_arguments(Seq2SeqTrainingArguments, "training_args")
    parser.add_class_arguments(EarlyStoppingCallback, "early_stopping")
    parser.add_argument("--generic.tgt_lang", required=True, help="souce language")
    parser.add_argument("--generic.src_lang", required=True)
    parser.add_argument("--generic.dataset", required=True, metavar="FILE", help="path to model file for bsd is '/home/sumire/discourse_context_mt/data/BSD-master/'")
    parser.add_argument("--generic.base_src_context",type=int, default="src", help="the number of the target context sentence for each input")
    parser.add_argument("--generic.base_tgt_context", type=int, default=0, help="the number of the source context sentence for each input")
    parser.add_argument("--generic.base_dropout", type=float, choices=np.arange(0.0, 1.0, 0.1), default=0, help="the coword dropout rate")
    parser.add_argument("--generic.base_tgt_sep", type=bool, default=False)# SU: changed default = True since error: pyarrow.lib.ArrowInvalid: Column 3 named context_ids expected length 70 but got length 1
    parser.add_argument("--generic.base_speaker", type=bool, default=False)
    parser.add_argument("--generic.context_speaker", type=bool, default=False)
    parser.add_argument("--generic.random_context", type=bool, default=False)
    parser.add_argument("--generic.base_tag", type=bool, default=False)
    parser.add_argument("--generic.context_tag", type=bool, default=False)
    parser.add_argument("--generic.cxmi", type=bool, default=True)
    parser.add_argument("--generic.context_src_context",type=int, default="src", help="the number of the target context sentence for each input")
    parser.add_argument("--generic.context_tgt_context", type=int, default=0, help="the number of the source context sentence for each input")
    parser.add_argument("--generic.context_dropout", type=float, choices=np.arange(0.0, 1.0, 0.1), default=0, help="the coword dropout rate")
    parser.add_argument("--generic.context_tgt_sep", type=bool, default=False)# SU: changed default = True since error: pyarrow.lib.ArrowInvalid: Column 3 named context_ids expected length 70 but got length 1
    parser.add_argument("--generic.checkpoint", required=True, metavar="FILE", help="path to best checkpoing for cxmi ") 
    return parser

def initialize_trainer(configs) -> TrainingArguments:
    
    trainer_args = Seq2SeqTrainingArguments(**namespace_to_dict(configs.training_args))
    early_stop_callback = EarlyStoppingCallback(
        **namespace_to_dict(configs.early_stopping)
    )
    callbacks = [early_stop_callback]

    return trainer_args, callbacks

def pred_prob_dist(model_type):
    parser = read_arguments()
     
    cfg = parser.parse_args()
    print(cfg)

    if model_type == "base":
        src_context_size = cfg.generic.base_src_context
        tgt_context_size = cfg.generic.base_tgt_context
        cw_dropout_rate = cfg.generic.base_dropout
        tgt_sep = cfg.generic.base_tgt_sep
        tag = cfg.generic.base_tag
        speaker = cfg.generic.base_speaker

    elif model_type == "context":
        src_context_size = cfg.generic.context_src_context
        tgt_context_size = cfg.generic.context_tgt_context
        cw_dropout_rate = cfg.generic.context_dropout
        tgt_sep = cfg.generic.context_tgt_sep
        tag = cfg.generic.context_tag
        speaker = cfg.generic.context_speaker

    src_lang = cfg.generic.src_lang
    tgt_lang = cfg.generic.tgt_lang
    file_path = cfg.generic.dataset
    #speaker = cfg.generic.speaker
    random_context = cfg.generic.random_context
    output_dir = cfg.training_args.output_dir
    #tag = cfg.generic.tag
    model_checkpoint = cfg.generic.checkpoint

    # Model for CXMI
    #### Choose One#######
    # For Src model
    #model_checkpoint = "/mnt/data-poseidon/sumire/bsd_en-ja/newest_truncate_padding_mex_length/cxmi/random_5-1/checkpoint-10000"
    
    # For Tgt model 

    # For Src and Tgt model
    #configuration = MBartConfig
    tokenizer = MBart50Tokenizer.from_pretrained(model_checkpoint, src_lang=f"{src_lang}_XX", tgt_lang=f"{tgt_lang}_XX")
    model = MBartForConditionalGenerationC.from_pretrained(model_checkpoint)

    # Add special token to separate context and current sentence
    tokenizer.add_special_tokens({"sep_token":"</t>"})

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

    honorifics = ["ござい", "ます", "いらっしゃれ", "いらっしゃい", "ご覧", "伺い", "伺っ", "存知", "です", "まし"]
    hon_id = tokenizer.encode(honorifics)
    #print ("speacial_tokens_dict", special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # Load the test dataset for CXMI
    file_path = file_path
    data_files = {"test": f"{file_path}test.json"}
    dataset = load_dataset("json", data_files=data_files)

    # Apply the preprocess function for the entire dataset 
    tokenized_datasets = dataset.map(
    partial(preprocess_2.preprocess_function, src_lang, tgt_lang, tag, speaker, src_context_size, tgt_context_size, random_context, 0.0, tgt_sep, tokenizer),
    batched=True,
    remove_columns=dataset["test"].column_names, # train
    )

    # Create a batch using DataCollator and pad dinamically
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt") 
    
    training_args, callbacks = initialize_trainer(cfg)

    trainer = Seq2SeqTrainerC(
        model=model,                         
        args=training_args,                  
        #train_dataset=tokenized_datasets["train"],        
        #eval_dataset=tokenized_datasets["test"],            
        data_collator=data_collator,
        tokenizer=tokenizer,
        #compute_metrics=partial(eval_bleu.compute_metrics, output_dir, tgt_lang, tokenizer),
        callbacks = callbacks, # Put tensorboard logger: [EarlyStoppingCallback(early_stopping_patience=10) , CustomLoggerCallback]
        #callbacks = integrations.TensorBoardCallback # for tensorboard call backs, don't know how to run this

        )
  
    model.eval()
    test_data = tokenized_datasets["test"]
    gold_labels = test_data["labels"]

    # Make test data into numpy from list
    """
    torch_ds = test_data.with_format("torch")
    test_loader = DataLoader(torch_ds, batch_size=16, shuffle=False)
    preds, _, _ = trainer.prediction_loop(test_loader, description="prediction")
    
    """
    preds, label_ids, metrics = trainer.predict(tokenized_datasets["test"])
    
    prob_dist = preds[0] # The second element of the predictions are hidden states

    print (type(prob_dist))
    print ("prob_dist", prob_dist)

    print ("shape", prob_dist.shape)
    # prob_dist : num_sent * seqlen * vocab 
    return gold_labels, prob_dist, hon_id

def sent_scores(gold_labels, prob_dist, hon_id): 
    # Skip token 1 ()
    # prob_dist : Instance x S x V 
    # gold_word_ids : Instance x S
    
    #softmax = nn.Softmax(dim=-1)
    softmax = nn.LogSoftmax(dim=-1)
    #prob_dist = F.log_softmax(torch.from_numpy(prob_dist), dim=-1)
    prob_dist = softmax(torch.from_numpy(prob_dist))
    all_sent_scores = [] # B x S
    all_hon_scores = []
    
    num_sents = prob_dist.shape[0] # slice to make it integer from tuple
    print ("Num of Instances", num_sents)
    for i in range(num_sents):
        scores = [] # S
        hon_scores = []
        print ("num_sents", i)
        seq_len = prob_dist.shape[1]
        #print ("seq_len", seq_len)
        for j in range(seq_len):
            # get probability of gold word
            gold_word_id = np.array(gold_labels)[i, j]
            
            if gold_word_id in hon_id:
                hon_gold_id = np.array(gold_labels)[i, j]
                hon_scores.append(prob_dist[i, j, hon_gold_id])
            #print (j)
            #if gold_word_id <= len(gold_labels[i]):
                #print (len(gold_labels[i]))
            if gold_word_id != 1: # pad token 
            #scores.append(argmax(all_prob_dist[i, j, :]))
                scores.append(prob_dist[i, j, gold_word_id])
        hon_sent_scores = sum(hon_scores)
        sent_scores = sum(scores)

        all_hon_scores.append(hon_sent_scores)
        all_sent_scores.append(sent_scores)
        
    return all_sent_scores, num_sents, all_hon_scores # B x S

def cxmi():
    # base_prob_list : sent_size 
    gold_labels, base_prob_dist, hon_id = pred_prob_dist(model_type="base")
    gpld_labels, context_prob_dist, hon_id = pred_prob_dist(model_type="context")

    base_sent_scores, base_num_sents, base_hon_scores = sent_scores(gold_labels, base_prob_dist, hon_id)
    context_sent_scores, context_num_sents, context_hon_scores = sent_scores(gold_labels, context_prob_dist, hon_id)

    
    cxmi = - (np.mean(np.array(base_sent_scores) - np.array(context_sent_scores)))
    hon_cxmi = - (np.mean(np.array(base_hon_scores) - np.array(context_hon_scores)))
    
    return cxmi, base_num_sents, context_num_sents, hon_cxmi

def main():
    cxmi_score, base_num_sents, context_num_sents, hon_cxmi_score = cxmi()

    #print ("cxmi shape", cxmi_score.shape)
    print (f"CXMI: {cxmi_score}")
    print (f"number of context model sentences:  {context_num_sents}", f"number of base model sentences:  {base_num_sents}")
    print (f"Honorific CXMI: {hon_cxmi_score}")
if __name__ == "__main__":
    main()
