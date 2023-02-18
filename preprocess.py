from datasets import load_dataset, concatenate_datasets # Huggingface datasets
import transformers
from transformers import MBart50TokenizerFast, MBart50Tokenizer, MBartConfig, MBartForConditionalGeneration, DataCollatorForSeq2Seq
import numpy as np
from functools import partial
import torch
import random

max_length = 128 # Should be modified considering bsd's max input size is 278 (en) and 110 (en) , but ami's max input size is 662 (en) and 302 (ja)
# 520 # 128

def preprocess_function(src_context_size, tgt_context_size, cw_dropout_rate, tgt_sep, tokenizer, data): # data should be splitted into train / dev / test internally
    inputs = []
    targets = []
    new_inputs = []
    new_targets = []
    new_tgt_contexts = []

    for doc in data["conversation"]:
        doc_input = [sent['en_sentence'] for sent in doc] # Iterate over every documents #list[doc[sent]] # for doc in docs. 
        doc_target = [sent['ja_sentence'] for sent in doc]
        inputs.append(doc_input)
        targets.append(doc_target)

        new_doc_input = []
        for idx, ip in enumerate (doc_input):
            src_context_size = random.randint(0, src_context_size) ####################Deal with when the context size is zero ###############
            if src_context_size == 0:
                new_doc_input.append(ip)
    
        # Context_aware inputs, "new_inputs"
            else:    
            # Check each inputs 
                context_list = []  
                # Check each context index given the context size and current input index
                for context_window in range(src_context_size, 0, -1):
                    context_idx = idx - context_window
                    
                    # If context idx is not the left side of the beggining of the inputs
                    if context_idx >= 0:
                        #Store the context in a list
                        context_list.append(doc_input[context_idx])
                    
                if len(context_list) ==0:
                    new_doc_input.append(ip)
                    
                else:
                    concat_contexts = "</t>".join(context_list)
                    #print (concat_contexts)

                    new_input = "</t>".join([concat_contexts,ip])
                    #print ("new_input", new_input)
                    new_doc_input.append(new_input)
                    #print ("new_doc_input", new_doc_input) 
        new_inputs.append(new_doc_input)
        #print ("-------")
        #print ("new_inputs_per doc-level:", new_inputs)
        #print ("-------")
        #print("new_inputs1", new_inputs)

        # Concatenate contexts given any context_size
        new_doc_target = []
        # Separate context and current sentences per doc
        new_doc_context = [] #
        for idx, tgt in enumerate (doc_target):
            tgt_context_size = random.randint(0, tgt_context_size) ############## Deal with when the context size is zero ############3
            if tgt_context_size == 0:
                new_doc_target.append(tgt) 
                new_doc_context.append("</t>") # to make context_ids the same shape with input_ids 
            else:
                tgt_context_list = []
                
                # Check each context index given the context size and current input index
                for context_window in range(tgt_context_size, 0, -1):
                    context_idx = idx - context_window
                    
                    # If context idx is not the left side of the beggining of the inputs
                    if context_idx >= 0:
                        #Store the context in a list
                        tgt_context_list.append(doc_target[context_idx])
                    
                if len(tgt_context_list) ==0:
                    new_doc_context.append("</t>")
                    new_doc_target.append(tgt)
                    
                else:
                    concat_contexts = "</t>".join(tgt_context_list)
                    #print (concat_contexts)
                    if tgt_sep : #
                        #print ("------------")
                        new_doc_context.append(concat_contexts)
                        print ("new_doc_context1", new_doc_context)
                        new_doc_target.append(tgt)
                        #print ("new_doc_target", new_doc_target)
                    else:
                        new_target = "</t>".join([concat_contexts,tgt])
                        new_doc_target.append(new_target) #
        
        new_tgt_contexts.append(new_doc_context) #     
        new_targets.append(new_doc_target)
        #print ("new_targets1", new_targets[:10])

    new_tgt_contexts = [sent for doc in new_tgt_contexts for sent in doc]
    print ("new_tgt_contexts[:10]", new_tgt_contexts[:10])
    context_out  = tokenizer(new_tgt_contexts, max_length=max_length,  truncation=False, padding=True) # Modify False
    context_ids = context_out['input_ids'] #
    context_attn = context_out['attention_mask'] #
    print ("context_ids", context_ids)
    new_inputs = [sent for doc in new_inputs for sent in doc]  
    print ("new_inputs[30:40]", new_inputs[30:40])  
    new_targets = [sent for doc in new_targets for sent in doc] 
    print ("new_targets[30:40]", new_targets[30:40] )    

    model_inputs = tokenizer(
            new_inputs, text_target=new_targets, max_length=max_length, truncation=False, padding=True
        )

    if tgt_context_size>0:
        ilen = len(context_ids[0])
        for ic, con in enumerate(context_ids): ####How does this work?"#####
            assert ilen == len(con), "line is  {0}".format(con)
        
        model_inputs['context_ids']=context_ids
        model_inputs['context_attention_mask']=context_attn
        print ('model_inputs["context_ids"]', model_inputs["context_ids"])

    if cw_dropout_rate>0:
        #new_input_ids = []
        for i,inst in enumerate(model_inputs['input_ids']):
            sep_indices = [i for i, x in enumerate(inst) if x == tokenizer.convert_tokens_to_ids("</t>")]
            eos_indices = [i for i, x in enumerate(inst) if x == tokenizer.eos_token_id] #should be only 1
            # print(sep_indices)
            # print(eos_indices)
            # print('----')
            if len(sep_indices)>0:
                sub = inst[sep_indices[-1]:eos_indices[-1]] # get the indices of the input to be modified (source)
                sub = list(range(sep_indices[-1],eos_indices[-1]))
                masks = np.random.choice([True, False], size=len(sub), p=[cw_dropout_rate, 1-cw_dropout_rate])
                to_be_masked = list(np.array(sub)[masks])
                for m in to_be_masked:
                    model_inputs['input_ids'][i][m] = tokenizer.mask_token_id

        print ("\nDecoded tokinized input-ids: ", tokenizer.batch_decode(model_inputs['input_ids'][30:40], skip_special_tokens=True))
        print ("\nDecoded tokinized labels: ", tokenizer.batch_decode(model_inputs['labels'][30:40], skip_special_tokens=True))
        #print ("\nDecoded tokinized context_ids: ", tokenizer.batch_decode(model_inputs["context_ids"][0:5], skip_special_tokens=True))
        #print ("\nDecoded tokinized context_atten: ", tokenizer.batch_decode(model_inputs['context_attention_mask'][30:40],skip_special_tokens=True))
    return model_inputs


# To make sure those below runs only when "python preprocess.py"
if __name__ == "__main__":
    # Load the dataset
    file_path = '/home/sumire/discourse_context_mt/data/BSD-master/'
    data_files = {"train": f"{file_path}short_train.json", "validation": f"{file_path}dev.json", "test": f"{file_path}short_test.json"}
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
    preprocess_function(4, 4, cw_dropout_rate, tgt_sep, tokenizer, dataset['train'])
    
    
