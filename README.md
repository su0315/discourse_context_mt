# Context-aware Neural Machine Translation for English-Japanese Business Scene Dialogues

This Repository contains the implementation of the training and evaluation of Context-aware Neural Machine Translation for English-Japanese Business Scene Dialogues.

Further information is here: 

The environment can be created by the yaml file.
```
conda env create -f environment.yml
conda activate context_aware_mt
```

In the config folder, the configuration files of all types of context-aware models that you can run with the command below.  

To train and evaluate the model, run
```
python main.py  --cfg MODEL_CONFIG_FILE
```
For example,
to train and evaluate context-agnostic model (1-1 model), run
```
python main.py  --cfg /path/to/config_1-1.yaml
```
to train and evaluate source side context model (e.g. 5-1 model), run
```
python main.py  --cfg /path/to/config_5-1.yaml
```
to compute CXMI and Honorifics CXMI score (e.g. between 5-1 model and 1-1 model), run
```
python cxmi.py  --cfg /path/to/cxmi_5-1.yaml
```
