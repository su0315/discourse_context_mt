from datasets import load_dataset, concatenate_datasets # Huggingface datasets
import transformers
from transformers import MBart50TokenizerFast, MBart50Tokenizer, MBartConfig, MBartForConditionalGeneration, DataCollatorForSeq2Seq
import numpy as np
from functools import partial
import torch
import random

max_length = 128 # Should be modified considering bsd's max input size is 278 (en) and 110 (en) , but ami's max input size is 662 (en) and 302 (ja)
# 520 # 128 #256
# random5_5 was 256


def preprocess_function(src_lang, tgt_lang, tag, speaker, src_context_size, tgt_context_size, random_context, cw_dropout_rate, tgt_sep, tokenizer, data): 
    new_inputs = []
    new_targets = []
    new_tgt_contexts = []

    if tag:
        scene_tags = data["tag"]
        # scene tags are: '<face-to-face conversation>', '<phone call>', '<general chatting>', '<meeting>', '<training>' and '<presentation>'

        for idx, tag in enumerate(scene_tags):
            scene_tags[idx] = f"<{scene_tags[idx]}>"

    # Iterate over every documents
    for doc_idx, doc in enumerate(data["conversation"]):
        doc_input = [sent[f'{src_lang}_sentence'] for sent in doc] 
        doc_target = [sent[f'{tgt_lang}_sentence'] for sent in doc]

        if speaker:
            src_speakers =  [sent[f'{src_lang}_speaker'] for sent in doc] 
        
        # Concatenate contexts given any context_size both in src and tgt
        # Source side
        new_doc_input = []
        
        for idx, ip in enumerate (doc_input):
            if speaker: 
                current_speaker = src_speakers[idx]
                #print ("current_speaker", current_speaker)

            if random_context:
                src_context_size = random.randint(0, src_context_size) 
                
                #randoms.append(src_context_size)
            
            if src_context_size == 0:
                if tag:
                    #print (f"{scene_tags[doc_idx]}{ip}")
                    new_doc_input.append(f"{scene_tags[doc_idx]}{ip}")
                    
                else:    
                    new_doc_input.append(ip)
    
            else:    
                context_list = []  
                # Check each context index given the context size and current input index
                for context_window in range(src_context_size, 0, -1):
                    context_idx = idx - context_window
                    
                    # If context idx is not the left side of the beggining of the doc_inputs
                    if context_idx >= 0:
                        context_sent = doc_input[context_idx]
                        if speaker:
                            context_speaker = src_speakers[context_idx]
                            if context_speaker != current_speaker:
                                context_sent = f"<DiffSpeak>{context_sent}"
                                #print ("concat_speaker", concat_speaker)
                                
                            else:
                                context_sent = f"<CurrSpeak>{context_sent}"
                                #print ("concat_speaker", concat_speaker)
                        
                        #Store the context in a list
                        context_list.append(context_sent)

                #print ("context_list", context_list)
                    
                if len(context_list) ==0:
                    sent = ip
                    if speaker or tag:
                        if not tag:
                            sent = f"<CurrSpeak>{sent}"
                        #print ("concat_speaker2:", concat_speaker)
                        if not speaker:
                            sent = f"{scene_tags[doc_idx]}{sent}"
                        
                        if tag and speaker:
                            sent = f"{scene_tags[doc_idx]}<CurrSpeak>{sent}"
                        
                    new_doc_input.append(sent)
                    
                else:
                    concat_contexts = "</t>".join(context_list)
                    sent = ip
                    if speaker or tag:
                        if not tag:
                            sent = f"<CurrSpeak>{sent}"
                            #new_input = "</t>".join([concat_contexts,sent])
                        if not speaker:
                            concat_contexts = f"{scene_tags[doc_idx]}{concat_contexts}"
                            #new_input = "</t>".join([concat_contexts,sent])
                        if tag and speaker:
                            concat_contexts = f"{scene_tags[doc_idx]}{concat_contexts}"
                            sent = f"<CurrSpeak>{sent}"
                    new_input = "</t>".join([concat_contexts,sent])
                    new_doc_input.append(new_input)
                    
        #print ("randoms:", randoms)   
        new_inputs.append(new_doc_input)
        
        # Target side
        new_doc_target = []
        # Separate context and current sentences per doc
        new_doc_context = [] 

        for idx, tgt in enumerate (doc_target):
            target_context_size = tgt_context_size

            if random_context:
                target_context_size = random.randint(0, tgt_context_size)
                
            if target_context_size == 0:
                new_doc_target.append(tgt) 
                new_doc_context.append("</t>") # SU: to make context_ids the same shape with input_ids 
            else:
                tgt_context_list = []
                
                # Check each context index given the context size and current target index
                for context_window in range(target_context_size, 0, -1):
                    context_idx = idx - context_window
                    
                    # If context idx is not the left side of the beggining of the inputs
                    if context_idx >= 0:
                        # Store the context in a list
                        tgt_context_list.append(doc_target[context_idx])
                
                # When there is no context, store </t> in context
                if len(tgt_context_list) ==0:
                    new_doc_context.append("</t>") # SU: when there is no context still there is contexxt_ids
                    new_doc_target.append(tgt)

                # When there are contects, concatenate and append to the context list 
                else:
                    concat_contexts = "</t>".join(tgt_context_list)
                    concat_contexts += "</t>" # After contexts, add sep token
        
                    if tgt_sep : 
                        new_doc_context.append(concat_contexts)
                        new_doc_target.append(tgt)
                        #print ("new_doc_target", new_doc_target)
                    else:
                        new_target = "</t>".join([concat_contexts,tgt])
                        new_doc_target.append(new_target) #

        # Collect the document level target to a list
        new_tgt_contexts.append(new_doc_context) 
        new_targets.append(new_doc_target)

    # Extract sentence level context in a list
    new_inputs = [sent for doc in new_inputs for sent in doc]
    new_targets = [sent for doc in new_targets for sent in doc] 
    new_tgt_contexts = [sent for doc in new_tgt_contexts for sent in doc]

    # Tokenize input and target ###########Now experimenting##########################
    
    model_inputs = tokenizer(
            new_inputs, text_target=new_targets, truncation=True,  max_length=max_length, padding = "max_length" ) # "max_length", truncation_side="left" #max_length=max_length,
    """
    # Old Setting Without Max length 
    # model_inputs = tokenizer(
            new_inputs, text_target=new_targets, truncation=False, padding = True )
    """
    # Tokenize context indipendently
    if tgt_context_size>0 or random_context: 
        #print ("new_tgt_contexts", new_tgt_contexts[:5])#SU
        
        context_out  = tokenizer(new_tgt_contexts,  truncation=True,  max_length=max_length, padding = "max_length" ) ### SU　"max_length", truncation_side="left", max_length=max_length,
        """
        # Old Setting Without Max length 
        # context_out  = tokenizer(new_tgt_contexts,  truncation=False, padding = True)
        """
        context_ids = context_out['input_ids'] ### SU
        context_attn = context_out['attention_mask'] ### SU
        
        # Add tokenized context information on model_inputs
        model_inputs['context_ids']=context_ids
        #print ("model_inputs['context_ids']", model_inputs['context_ids'][10:20])
        model_inputs['context_attention_mask']=context_attn
        #print ("model_inputs['context_attention_mask']", model_inputs['context_attention_mask'][:5])
        #print ("model_inputs['input_ids']", model_inputs["input_ids"][:5])

        #print ("\nDecoded tokenized context_ids: ", tokenizer.batch_decode(model_inputs["context_ids"][0:10], skip_special_tokens=True))
        #print ("\nDecoded tokenized context_atten: ", tokenizer.batch_decode(model_inputs['context_attention_mask'][0:10],skip_special_tokens=False))
       
    if cw_dropout_rate>0:
        for i,inst in enumerate(model_inputs['input_ids']):
            sep_indices = [i for i, x in enumerate(inst) if x == tokenizer.convert_tokens_to_ids("</t>")]
            eos_indices = [i for i, x in enumerate(inst) if x == tokenizer.eos_token_id] #should be only 1
    
            if len(sep_indices)>0:
                sub = inst[sep_indices[-1]:eos_indices[-1]] # get the indices of the input to be modified (source)
                sub = list(range(sep_indices[-1],eos_indices[-1]))
                masks = np.random.choice([True, False], size=len(sub), p=[cw_dropout_rate, 1-cw_dropout_rate])
                to_be_masked = list(np.array(sub)[masks])
                for m in to_be_masked:
                    model_inputs['input_ids'][i][m] = tokenizer.mask_token_id

    print ("\nDecoded tokinized input-ids: ", tokenizer.batch_decode(model_inputs['input_ids'][:10], skip_special_tokens=False))
    print ("\nDecoded tokinized labels: ", tokenizer.batch_decode(model_inputs['labels'][0:10], skip_special_tokens=False))

    return model_inputs


# To make sure those below runs only when "python preprocess.py"
if __name__ == "__main__":
    # Load the dataset
    file_path = '/home/sumire/discourse_context_mt/data/BSD-master/'
    data_files = {"train": f"{file_path}short_train.json", "validation": f"{file_path}short_dev.json", "test": f"{file_path}short_test.json"}
    dataset = load_dataset("json", data_files=data_files)
    
    # Tokenize using "facebook/mbart-large-50-many-to-many-mmt"
    model_checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
    configuration = MBartConfig()
    source_lang="en"
    target_lang="ja"
    tokenizer = MBart50Tokenizer.from_pretrained(model_checkpoint, src_lang=f"{source_lang}_XX", tgt_lang=f"{target_lang}_XX")
    model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)
    # Add special token to separate context and current sentence
    tokenizer.add_special_tokens({"sep_token":"</t>"})
    model.resize_token_embeddings(len(tokenizer))
    cw_dropout_rate=0.2
    tgt_sep = True
    tag = True
    speaker = True
    random_context = False
    if speaker:
        special_tokens_dict = {'additional_special_tokens': ['<CurrSpeak>','<DiffSpeak>']}
        tokenizer.add_special_tokens(special_tokens_dict)

    if tag:
        special_tokens_dict = {'additional_special_tokens': ['<Face-to-Face>','<Phone call>', '<General chatting>', '<Meeting>', '<Training>', '<Presentation>']}
    
    model.resize_token_embeddings(len(tokenizer))
    
    #preprocess_function(4, 4, cw_dropout_rate, tgt_sep, tokenizer, dataset['train'])
    preprocess_function(source_lang, target_lang, tag, speaker, 4, 4, random_context, cw_dropout_rate, tgt_sep, tokenizer, dataset['train'])


    
