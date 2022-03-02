# EntLM 
The source codes for EntLM.

## Dependencies:

Cuda 10.1, python 3.6.5

To install the required packages by following commands:

```
$ pip3 install -r requirements.txt
```

To download the pretrained bert-base-cased model:
```
$ cd pretrained/bert-base-cased/
$ sh download_bert.sh
```

## Few-shot Experiment

Run the few-shot experiments on CoNLL 5-shot with:
```
sh scripts/run_conll.sh
```
By default, this runs 4 rounds of experiments for each of the sampled datasets.
You can also run 10/20/50-shot experiments by editing the line *FILE_PATH=dataset/conll/5shot/* in scripts/run_conll.sh .


## Label word selection

You can run the label word selection process by:
```
sh scripts/count_freq.sh
```
This will build a label_map file such as *dataset/conll/label_map_timesup_ratio0.6_multitoken_top6.json* in the dataset path.

You can try different method by changing *"--sort_method"* to *["LM", "data", "timesup"]*.

Or you can try different ratio/virtual_number by changing *"--filter_ratio"* and *"--top_k_num"*.