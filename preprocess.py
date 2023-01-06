from datasets import load_dataset, concatenate_datasets # Huggingface datasets
import transformers
from transformers import MBart50Tokenizer, MBartConfig, MBartForConditionalGeneration, DataCollatorForSeq2Seq

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

def preprocess_function(data): # data should be splitted into train / dev / test internally
    inputs = [sent['en_sentence'] for doc in data["conversation"] for sent in doc]
    targets = [sent['ja_sentence'] for doc in data["conversation"] for sent in doc]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs

# Apply the preprocess function for the entire dataset 
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# Create a batch using DataCollator and pad dinamically
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt") 
batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])