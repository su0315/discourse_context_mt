# Context-aware Neural Machine Translation for English-Japanese Business Scene Dialogues

This Repository contains the implementation of the training and evaluation of Context-aware Neural Machine Translation for English-Japanese Business Scene Dialogues.

Further information can be found in ``IM_report_honda.pdf``.

The environment can be created by the yaml file.
```
conda env create -f environment.yml
conda activate context_aware_mt
```

In the config folder, the configuration files of all types of context-aware models that you can run with the command below.  

To train and evaluate the model, run
```
cd src
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
Hyperparameters used for the experiment is below.
In this experiment, truncation is not set so that the model does not cut the current sentence when the context size becomes larger. 
All of the parameters and a more detailed setup are in configuration files. 
| Parameter  | Value |
| ------------- | ------------- |
|Max Input Size for Padding | 128 | 
|Batch Size | 4 |
|Learning Rate | 2e-5 |
|Warmup Steps | 500 |
|Weight Decay | 0.01 |
|Train Epochs | 5 (10 for CXMI random models) |
|Early Stopping Patience | 3 (5 for CXMI random models) |
