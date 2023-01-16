from datasets import load_dataset, concatenate_datasets # Huggingface datasets
import transformers
from transformers import MBart50Tokenizer, MBartConfig, MBartForConditionalGeneration, DataCollatorForSeq2Seq

from functools import partial

# Load the dataset
file_path = "/home/sumire/discourse_context_mt/data/BSD-master/"
data_files = {"train": f"{file_path}train.json", "validation": f"{file_path}dev.json", "test": f"{file_path}test.json"}
dataset = load_dataset("json", data_files=data_files)

# Tokenize using "facebook/mbart-large-50-many-to-many-mmt"
model_checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
configuration = MBartConfig()
tokenizer = MBart50Tokenizer.from_pretrained(model_checkpoint, src_lang="en_XX", tgt_lang="ja_XX")
model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)

max_length = 128


def preprocess_function(data, context_size): # data should be splitted into train / dev / test internally
    inputs = [sent['en_sentence'] for doc in data["conversation"] for sent in doc]
    targets = [sent['ja_sentence'] for doc in data["conversation"] for sent in doc]
    context_size = 1

    if context_size == 0:
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=max_length, truncation=True
        )

    # Context_aware inputs, "new_inputs"
    elif context_size == 1:
        new_inputs = []

        for idx, input in enumerate(inputs):
            if idx-context_size < 0: # Maybr only train dataset's beggining is marked
            
                #print (idx)
                new_input = f"<>{input}"
                #print (new_input)
                new_inputs.append(new_input)

            elif 0 <= idx-context_size:
                #print (idx)
                new_input = f"{inputs[idx-1]}</s>{input}"
                #print (new_input)
                new_inputs.append(new_input)
        #print ("The first 5 contextual inputs:", new_inputs[:5])
        #print ("The last 5 contextual inputs:", new_inputs[-5:])
        #print (tokenizer.batch_decode( new_inputs[:5] , skip_special_tokens=False))

        # Modify below inputs or new_inputs depending on the context-aware or not
        model_inputs = tokenizer(
            new_inputs, text_target=targets, max_length=max_length, truncation=True
        )
    return model_inputs

#model_inputs = preprocess_function(dataset['train'])
#print (model_inputs['input_ids'])

# Apply the preprocess function for the entire dataset 
tokenized_datasets = dataset.map(
    partial(preprocess_function, context_size=1),
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# Check if the inputs are really context aware 
#print (tokenized_datasets["train"][:5])
model_inputs = preprocess_function(dataset['validation'], context_size=1)
#print ("The first 5 contextual input-ids:", model_inputs["input_ids"][:5])
#print ("The last 5 contextual input-ids:", model_inputs["input_ids"][-5:])
#print (tokenizer.decode(model_inputs['input_ids'][1]))
#print (tokenizer.decode(tokenized_datasets["train"][:5]['input_ids'][3]))
#print (tokenized_datasets["train"][:5]['input_ids'][3])
#print (tokenizer.decode(tokenized_datasets["train"][:5]['input_ids'][4]))
#print (tokenized_datasets["train"][:5]['input_ids'][4])
#print (tokenized_datasets["train"][:6]['input_ids'][5])
#print (tokenizer.decode(tokenized_datasets["train"][:5]['input_ids'][4]))
#print (tokenizer.decode(tokenized_datasets["train"][:5]['labels'][3]))
#print (tokenizer.decode(tokenized_datasets["train"][:5]['labels'][4]))
#print (tokenizer.decode(tokenized_datasets["test"][:5]['input_ids'][0]))
#print (tokenizer.decode(tokenized_datasets["test"][:5]['input_ids'][1]))


# Create a batch using DataCollator and pad dinamically
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt") 
batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])

# Check the actual input characters
print (tokenizer.batch_decode(model_inputs["input_ids"][:5], skip_special_tokens=False))
print(tokenizer.batch_decode(tokenizer.build_inputs_with_special_tokens(model_inputs["input_ids"][:5])))