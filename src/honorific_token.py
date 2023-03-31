import transformers
from transformers import  MBart50Tokenizer, MBartConfig, MBartForConditionalGeneration
import pandas as pd
import numpy as np


model_checkpoint = "/mnt/data-poseidon/sumire/bsd_en-ja/newest_truncate_padding_mex_length/src_attention/3-1-t_sp_sc/checkpoint-10000"
tokenizer = MBart50Tokenizer.from_pretrained(model_checkpoint, src_lang="ja_XX", tgt_lang="en_XX")
model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)

#hon_id = tokenizer.encode(["ございます", "ます", "いらっしゃれば", "いらっしゃいますので", "ご覧", "伺います", "伺った", "存知ます", "です", "ました", "まして"])
hon_id = tokenizer.encode(["です", "でした", "ます","ました","ません","ましょう","でしょう","ください","ございます","おります", "致します", "ご覧", "なります", "伺", "頂く", "頂き", "頂いて", "下さい", "申し上げます"])
print (hon_id)
#[250012, 1453, 11593, 5574, 6465, 22245, 39256, 12547, 10712, 95689, 55826, 118916, 111521, 100519, 172974, 147406, 108780, 142996, 24443, 148852, 2]

hon_id.remove(tokenizer.lang_code_to_id["ja_XX"])
hon_id.remove(tokenizer.eos_token_id)
#print (hon_id)
for i in hon_id:
    token = tokenizer.decode(i)
    print (token)

print (tokenizer.encode("でしょう"))


#[250004, 3, 5574, 3, 3, 111521, 3, 3, 3, 1453, 3, 2]
sent_list = []
test_df = pd.read_json("/home/sumire/discourse_context_mt/data/BSD-master/test_cxmi2.json")
for doc in test_df["conversation"]:
    for sent in doc:
        ja_sent = sent["ja_sentence"]
        sent_list.append(ja_sent)

print (sent_list[502])
    
"""
with open('/home/sumire/discourse_context_mt/data/BSD-master/test_sent.txt','w', encoding='utf8') as wf:
    for doc in test_df["conversation"]:
        for sent in doc:
            ja_sent = sent["ja_sentence"]
            sent_list.append(ja_sent)    
            wf.write(f"{ja_sent}\n")
        #wf.write(f"Honorific CXMI: {hon_cxmi}")
"""


        
