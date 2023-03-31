import pandas as pd
import numpy as np

df_train = pd.read_json("/home/sumire/discourse_context_mt/data/BSD-master/train.json")
df_dev = pd.read_json("/home/sumire/discourse_context_mt/data/BSD-master/dev.json")
df_test = pd.read_json("/home/sumire/discourse_context_mt/data/BSD-master/test.json")

# make new df with id and mono-lingual sentence
# Japanese
#df_new = df_train.drop(['tag', 'title', 'original_language'], axis=1 )

def docid_generate(path, pathout):
    sent_list = []
    sent_no_list = []
    id_list = []

    for i in range(len(df_train['conversation'])):
        #print (len(conv))
        #print (df_train.index[df_train['conversation']==df_train['conversation'].iloc[i]])
        #df.index[df['column_name']==value].tolist()
        #conv_id = df_train['id'].iloc[conv.index]
        for sentence in df_train['conversation'].iloc[i]:
            id_list.append(df_train['id'].iloc[i])
            ja_sent_list.append(sentence['ja_sentence'])
            ja_sent_no_list.append(sentence['no'])
            
    print (len(id_list), len(ja_sent_list), len(ja_sent_no_list))

ja_train_id_sentence_df = pd.concat([pd.DataFrame(id_list, columns = ['id']), pd.DataFrame(ja_sent_no_list, columns = ['sentence_no']), pd.DataFrame(ja_sent_list, columns = ['sentence'])],axis=1)
ja_train_id_sentence_df