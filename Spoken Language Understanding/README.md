### About the task

Spoken language semantic understanding is an important part of the task-based dialogue system. Its purpose is to extract the user's intention from the user's utterance, provide information for the subsequent modules, and finally successfully complete the user's task. In general, we use several semantic units to describe the user's semantics, and each semantic unit consists of three parts: act, slot, and value.


### Data

+ `train.json`: train_dataset
+ `development.json`:dev_dataset
+ `ontology.json`:Ontology, including all involved actions, semantic slots and slot values
+ `lexicon`:The slot values corresponding to some semantic slots in the ontology are stored in this directory



### create the environment

    conda create -n slu python=3.6
    source activate slu
    pip install torch==1.8.1+cu102
    pip install transformers==4.23.0
    pip install xpinyin
    pip install Pinyin2Hanzi

### train baseline model 
    
run in the root directory

    python scripts/slu_baseline.py


### pre-train model
    
run in the root directory

    python scripts/bert_finetune.py

    python scripts/roberta_finetune.py

    python scripts/macbert_finetune.py

### test
    
run in the root directory

    python scripts/xx.py --testing # xx is the training scripts you used


results are in data/test.json
### inform

+ `utils/args.py`:defines all optional parameters involved
+ `utils/initialization.py`:Initialize system settings, including setting random seed and GPU/CPU
+ `utils/vocab.py`:Build a vocabulary for encoding input and output
+ `utils/word2vec.py`:turn words to vectors
+ `utils/example.py`:build dataset
+ `utils/batch.py`:build batch
+ `model/slu_bert.py`:pretain-model
+ `scripts/bert_finetune.py`:scripts for pretraining


