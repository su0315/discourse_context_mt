# Context-aware Neural Machine Translation for English-Japanese Business Scene Dialogues

This Repository contains the implementation of the training and evaluation of Context-aware Neural Machine Translation for English-Japanese Business Scene Dialogues.



The environment can be created by the yaml file.
```
conda env create -f environment.yml
conda activate context_aware_mt
```

In the config folder, the configuration files of all types of context-aware models that you can run with the command below.  

First, move to src directory
```
cd src
```
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
Hyperparameters used for the experiment is below. Optimizer is default to adam. 
All of the parameters and a more detailed setup are in configuration files. 
| Parameter  | Value |
| ------------- | ------------- |
|Max Input Size for Padding | 128 | 
|Truncation | True |
|Batch Size | 4 |
|Learning Rate | 2e-5 |
|Warmup Steps | 500 |
|Weight Decay | 0.01 |
|Train Epochs | 5 (10 for CXMI random models) |
|Early Stopping Patience | 3 (5 for CXMI random models) |
|Metric for bestmodel | comet|

---
## Reference
Please cite the following paper:
Sumire Honda, Patrick Fernandes, and Chrysoula Zerva. 2023. "[Context-aware Neural Machine Translation for English-Japanese Business Scene Dialogues](https://aclanthology.org/2023.mtsummit-research.23/)." In Proceedings of Machine Translation Summit XIX, Vol. 1: Research Track, pages 272–285, Macau SAR, China. Asia-Pacific Association for Machine Translation.

```bibtex
@inproceedings{honda-etal-2023-context,
    title = "Context-aware Neural Machine Translation for {E}nglish-{J}apanese Business Scene Dialogues",
    author = "Honda, Sumire  and
      Fernandes, Patrick  and
      Zerva, Chrysoula",
    editor = "Utiyama, Masao  and
      Wang, Rui",
    booktitle = "Proceedings of Machine Translation Summit XIX, Vol. 1: Research Track",
    month = sep,
    year = "2023",
    address = "Macau SAR, China",
    publisher = "Asia-Pacific Association for Machine Translation",
    url = "https://aclanthology.org/2023.mtsummit-research.23",
    pages = "272--285",
    abstract = "Despite the remarkable advancements in machine translation, the current sentence-level paradigm faces challenges when dealing with highly-contextual languages like Japanese. In this paper, we explore how context-awareness can improve the performance of the current Neural Machine Translation (NMT) models for English-Japanese business dialogues translation, and what kind of context provides meaningful information to improve translation. As business dialogue involves complex discourse phenomena but offers scarce training resources, we adapted a pretrained mBART model, finetuning on multi-sentence dialogue data, which allows us to experiment with different contexts. We investigate the impact of larger context sizes and propose novel context tokens encoding extra-sentential information, such as speaker turn and scene type. We make use of Conditional Cross-Mutual Information (CXMI) to explore how much of the context the model uses and generalise CXMI to study the impact of the extra sentential context. Overall, we find that models leverage both preceding sentences and extra-sentential context (with CXMI increasing with context size) and we provide a more focused analysis on honorifics translation. Regarding translation quality, increased source-side context paired with scene and speaker information improves the model performance compared to previous work and our context-agnostic baselines, measured in BLEU and COMET metrics.",
}
```

