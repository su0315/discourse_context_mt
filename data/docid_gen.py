path_test = '/home/sumire/main/contextual-mt/data/BSD-master/for_preprocess/bin/test.en-ja.en.idx'
pathout_test = '/home/sumire/main/contextual-mt/data/BSD-master/for_preprocess/bin/test.en-ja.docids'

path_val = '/home/sumire/main/contextual-mt/data/BSD-master/for_preprocess/bin/valid.en-ja.en.idx'
pathout_val = '/home/sumire/main/contextual-mt/data/BSD-master/for_preprocess/bin/valid.en-ja.en.docids'

path_train = '/home/sumire/main/contextual-mt/data/BSD-master/for_preprocess/bin/train.en-ja.en.idx'
pathout_train = '/home/sumire/main/contextual-mt/data/BSD-master/for_preprocess/bin/train.en-ja.en.docids'


def docid_generate(path, pathout):
    l = []
    with open (path,'r', encoding='utf-8') as f:
        for line in f.readlines():
            l.append(line.strip())

    docs =  dict.fromkeys(l)
    i = 0
    for d in docs:
        docs[d]=i 
        i+=1
        
    with open(pathout,'w', encoding='utf-8') as wf:
        for key in l:
            wf.write(str(docs[key])+'\n')

docid_generate(path_train, pathout_train)
docid_generate(path_val, pathout_val)
docid_generate(path_test, pathout_test)