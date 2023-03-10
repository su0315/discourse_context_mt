import evaluate # Huggingface evaluatetokenizer
import numpy as np
import preprocess_2
import json

metric1 = evaluate.load("sacrebleu")
metric2 =  evaluate.load("comet") # Added comet

def postprocess_text(preds, labels, input_ids):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    input_ids = [[input_id.strip()] for input_id in input_ids]

    return preds, labels, input_ids


def compute_metrics(output_dir, tgt_lang, tokenizer, eval_preds):
    preds, labels, input_ids = eval_preds # Check the location of input_ids is appropriate
    
    # Preds
    if isinstance(preds, tuple):
        preds = preds
    
    sep = tokenizer.sep_token_id
    preds = [ np.array_split(item, np.where(item == sep)[-1])[-1]  for item in preds ]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    #print ("decoded_preds: ", decoded_preds[:5])
    #with open('./results/bsd_en-ja/bleu_ja_pred/inference.json', 'w', encoding='utf8') as json_file:
        #json.dump(decoded_preds, json_file, ensure_ascii=False,)
    
    # Store inference
    with open(output_dir+'/translations.txt','w', encoding='utf8') as wf:
         for translation in decoded_preds:
            wf.write(translation.strip()+'\n') 

    #Labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels= [ np.array_split(item, np.where(item == sep)[-1])[-1]  for item in labels ]
    #print ("checking labels_token:")
    #print (labels[:10][:5])
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #print ("decoded_labels:", decoded_labels[:5])

    
    # Input_ids
    # For comet source info
    input_ids = np.where(input_ids != -100, input_ids, tokenizer.pad_token_id)
    #print ("checking input_ids before split:", input_ids[:10][:5])
    input_ids = [ np.array_split(item, np.where(item == sep)[-1])[-1]  for item in input_ids ]
    #print ("checking input_ids3 after split:")
    #print (input_ids[:10][:5])
    decoded_input_ids = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    

    decoded_preds, decoded_labels, decoded_input_ids = postprocess_text(decoded_preds, decoded_labels, decoded_input_ids)
    
    # bleu
    if tgt_lang == "ja":
        bleu = metric1.compute(predictions=decoded_preds, references=decoded_labels, tokenize='ja-mecab')
    else: 
        bleu = metric1.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": bleu["score"]}

    # comet
    print ("decoded_input_ids:",  [item for decoded_input_id in decoded_input_ids for item in decoded_input_id][:5], "\ndecoded_preds", decoded_preds[:5], "\ndecoded_labels", [item for decoded_label in decoded_labels for item in decoded_label][:5])
    
    comet = metric2.compute(predictions=decoded_preds, references=[item for decoded_label in decoded_labels for item in decoded_label], sources = [item for decoded_input_id in decoded_input_ids for item in decoded_input_id])
    result["comet"] =  np.mean(comet["scores"])
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    print(result)

    # Store the score
    with open(output_dir+'/test_score.txt','w', encoding='utf8') as wf:
        for key, value in result.items():
            wf.write(f"{key}: {value}\n") #ensure_ascii=False

    return result


