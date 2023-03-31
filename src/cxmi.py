import argparse
import preprocess, eval_bleu
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
    random_context = cfg.generic.random_context
    output_dir = cfg.training_args.output_dir
    model_checkpoint = cfg.generic.checkpoint
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

    # Add Scene Tags
    scene_tags = ['<face-to-face conversation>','<phone call>', '<general chatting>', '<meeting>', '<training>', '<presentation>']
    for i in scene_tags:
        special_tokens.append(i)
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    # Honorific Tokens to be used in Honorifics CXMI
    hon_id = tokenizer.encode(["です", "でした", "ます", "ました","ません","ましょう","でしょう","ください","ございます","おります", "致します", "ご覧", "なります", "伺", "頂く", "頂き", "頂いて", "下さい", "申し上げます"])
    hon_id.remove(tokenizer.eos_token_id)
    print (tokenizer.decode(hon_id))
    model.resize_token_embeddings(len(tokenizer))

    # Load the test dataset for CXMI
    file_path = file_path
    data_files = {"test": f"{file_path}test_cxmi1.json"}
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

    preds, label_ids, metrics = trainer.predict(tokenized_datasets["test"])
    
    prob_dist = preds[0] # The second element of the predictions are hidden states

    # Predict and store the probability distribution over the vocabularies
    decoded_preds = tokenizer.batch_decode(np.argmax(prob_dist, axis=-1), skip_special_tokens=True)

    with open(output_dir+'/translations.txt','w', encoding='utf8') as wf:
         for translation in decoded_preds:
            wf.write(translation.strip()+'\n') 

    return gold_labels, prob_dist, hon_id, output_dir, tokenizer

def sent_scores(gold_labels, prob_dist, hon_id): 
    
    """
    
    Given the probability distribution, calculate the sentence-wise probability score over the gold label vocabularies

    Args
        gold_labels: np.ndarray of gold label token in test set
        prob_dist: np.ndarray of probability distribution 
        hon_id : list

    Return
        all_sent_scores: np.ndarray of scores for all sentences 
        num_sents: Int 
        all_hon_scores: np.ndarray of honorifics scores for all honorific tokens 
        hon_id_list: list of the honorific token id occured in the test data

    """
    # prob_dist : Instance x S x V 
    # gold_word_ids : Instance x S
    
    softmax = nn.LogSoftmax(dim=-1)
    prob_dist = softmax(torch.from_numpy(prob_dist))
    all_sent_scores = [] # B x S
    all_hon_scores = []
    
    num_sents = prob_dist.shape[0] # slice to make it integer from tuple
    print ("Num of Instances", num_sents)
    hon_id_list = []
    for i in range(num_sents):
        scores = [] # S
        print ("num_sents", i)
        seq_len = prob_dist.shape[1]

        for j in range(seq_len):
            # get probability of gold word
            gold_word_id = np.array(gold_labels)[i, j]
            
            if gold_word_id in hon_id:
                hon_gold_id = np.array(gold_labels)[i, j]
                print ("honorific_gold_id", hon_gold_id)
                hon_loc_id = {f"Sentence{i} Word{j}":hon_gold_id}
                hon_id_list.append(hon_loc_id)
                all_hon_scores.append(prob_dist[i, j, hon_gold_id])

            if gold_word_id != 1: # pad token 
                scores.append(prob_dist[i, j, gold_word_id])
        sent_scores = sum(scores)
        all_sent_scores.append(sent_scores)
        
    print ("all_hon_scores", all_hon_scores)
    return all_sent_scores, num_sents, all_hon_scores, hon_id_list # B x S

def cxmi():
    # base_prob_list : sent_size 
    gold_labels, base_prob_dist, hon_id, output_dir, tokenizer = pred_prob_dist(model_type="base")
    gpld_labels, context_prob_dist, hon_id, output_dir, tokenizer = pred_prob_dist(model_type="context")

    base_sent_scores, base_num_sents, base_hon_scores, hon_id_list = sent_scores(gold_labels, base_prob_dist, hon_id)
    context_sent_scores, context_num_sents, context_hon_scores, hon_id_list = sent_scores(gold_labels, context_prob_dist, hon_id)

    cxmi_per_sent = np.array(base_sent_scores) - np.array(context_sent_scores)
    max_cxmi_sent_id = np.argmax(-cxmi_per_sent)
    max_cxmi_score = -cxmi_per_sent[max_cxmi_sent_id]
    cxmi = - (np.mean(cxmi_per_sent))
    hon_scores_per_hon_word = np.array(base_hon_scores) - np.array(context_hon_scores)
    print ("hon_scores", hon_scores_per_hon_word)
    argmax_hon_cxmi = np.argmax(-(hon_scores_per_hon_word))
    max_hon_cxmi = -hon_scores_per_hon_word[argmax_hon_cxmi]
    hon_cxmi = - (np.mean (hon_scores_per_hon_word))

    
    with open(output_dir+'/cxmi_score.txt','w', encoding='utf8') as wf:
        wf.write(f"CXMI: {cxmi}\nMax CXMI Sentence: Sent{max_cxmi_sent_id}: {max_cxmi_score}\nHonorific CXMI: {hon_cxmi}\n({len(hon_id_list)})Used Honorifics: {tokenizer.decode([v for dict in hon_id_list for v in dict.values()])}\nHonorific ID: {hon_id_list}\nMax Hon CXMI: The {argmax_hon_cxmi}rd used Honorific ({tokenizer.decode((hon_id_list[argmax_hon_cxmi]).values())}) / {max_hon_cxmi}\n All Honorific CXMI: {hon_scores_per_hon_word}") #ensure_ascii=False
        
    print (f"number of honorific words:  {len(hon_id_list)}")
    return cxmi, base_num_sents, context_num_sents, hon_cxmi

def main():
    cxmi_score, base_num_sents, context_num_sents, hon_cxmi_score = cxmi()

    print (f"CXMI: {cxmi_score}")
    print (f"number of context model sentences:  {context_num_sents}", f"number of base model sentences:  {base_num_sents}")
    print (f"Honorific CXMI: {hon_cxmi_score}")



if __name__ == "__main__":
    main()
