# discourse_context_mt

Updated at Jan 17th

To run the model with English-Japanese, BSD dataset and the context size 1,

run the script below in the discourse_context_mt directory
The "-p" might be adapted depending on where you store the BSD dataset

```
python main.py -s en -t ja -p '../discourse_context_mt/data/BSD-master/' -c 1
```
