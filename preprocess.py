from datasets import load_dataset, concatenate_datasets # Huggingface datasets
import transformers
from transformers import MBart50TokenizerFast, MBart50Tokenizer, MBartConfig, MBartForConditionalGeneration, DataCollatorForSeq2Seq

# Load the dataset
#file_path = '/home/sumire/discourse_context_mt/data/BSD-master/'
#data_files = {"train": f"{file_path}train.json", "validation": f"{file_path}dev.json", "test": f"{file_path}test.json"}
#dataset = load_dataset("json", data_files=data_files)

# Tokenize using "facebook/mbart-large-50-many-to-many-mmt"
#model_checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
#configuration = MBartConfig()
#source_lang="en"
#target_lang="ja"
#tokenizer = MBart50Tokenizer.from_pretrained(model_checkpoint, src_lang=f"{source_lang}_XX", tgt_lang=f"{target_lang}_XX")
#model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)

max_length = 128

def preprocess_function(context_size, tokenizer, data): # data should be splitted into train / dev / test internally
    
    #print (data)
    inputs = [sent['en_sentence'] for doc in data["conversation"] for sent in doc]
    targets = [sent['ja_sentence'] for doc in data["conversation"] for sent in doc]
    #print (context_size)

    if context_size == 0:
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=max_length, truncation=True
        )
        print ("\nDecoded tokinized input-ids: ", tokenizer.batch_decode(model_inputs['input_ids'][:10], skip_special_tokens=False))
        
    
    # Context_aware inputs, "new_inputs"
    else:
        
        """
        #Previous code for context_size = 1
        #print ("here")
        new_inputs = []
        for idx, input in enumerate(inputs):
           
            #new_input = f"{inputs[idx-1]}</s>{input}"
            new_input = f"{inputs[idx-1]}{tokenizer.sep_token}{input}"
            new_inputs.append(new_input)
        """
        # Concatenate contexts given any context_size
        new_inputs = []
        # Check each inputs 
        for idx, ip in enumerate (inputs):
            context_list = []
            
            # Check each context index given the context size and current input index
            for context_window in range(context_size, 0, -1):
                context_idx = idx - context_window
                
                # If context idx is not the left side of the beggining of the inputs
                if context_idx >= 0:
                    #Store the context in a list
                    context_list.append(inputs[context_idx])
                
            if len(context_list) ==0:
                new_inputs.append(ip)
                
            else:
                concat_contexts = "</s>".join(context_list)
                #print (concat_contexts)

                new_input = "</s>".join([concat_contexts,ip])
                #print (new_input)
                new_inputs.append(new_input)


        # Modify below inputs or new_inputs depending on the context-aware or not
        model_inputs = tokenizer(
            new_inputs, text_target=targets, max_length=max_length, truncation=True
        )

        # Check the actual input before / after tokenizer
        print ("\nInputs to be put in tokenizer: ", new_inputs[:10])
        print ("\nDecoded tokinized input-ids: ", tokenizer.batch_decode(model_inputs['input_ids'][:10], skip_special_tokens=False))
        #print ("model_inputs: ", tokenizer.decode(model_inputs['input_ids']))
    return model_inputs


# Apply the preprocess function for the entire dataset 
#tokenized_datasets = dataset.map(
    #preprocess_function,
    #batched=True,
    #remove_columns=dataset["train"].column_names,
#)

# Create a batch using DataCollator and pad dinamically
#data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt") 
#batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])

if __name__ == "__main__":
    # Load the dataset
    file_path = '/home/sumire/discourse_context_mt/data/BSD-master/'
    data_files = {"train": f"{file_path}train.json", "validation": f"{file_path}dev.json", "test": f"{file_path}test.json"}
    dataset = load_dataset("json", data_files=data_files)
    
    # Tokenize using "facebook/mbart-large-50-many-to-many-mmt"
    model_checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
    configuration = MBartConfig()
    source_lang="en"
    target_lang="ja"
    tokenizer = MBart50Tokenizer.from_pretrained(model_checkpoint, src_lang=f"{source_lang}_XX", tgt_lang=f"{target_lang}_XX")
    #
    
    model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)
    preprocess_function(3, tokenizer, dataset['validation'])
    #print(tokenizer.eos_token) 
    #print(tokenizer.sep_token_id)
    #print("test: ", tokenizer.decode([2]))
    #print(tokenizer.decode(tokenizer.sep_token_id))
