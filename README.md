# RFTokenizer
A character-wise tokenizer for morphologically rich languages

For replication of paper results see replication.md

For full NLP pipelines for morphologically rich languages (MRLs) based on this tool, see:

  * Coptic: http://www.github.com/CopticScriptorium/coptic-nlp/
  * Hebrew: http://www.github.com/amir-zeldes/HebPipe/
  
Pretrained models are provided for **Coptic**, **Arabic** and **Hebrew**

## Installation

RFTokenizer is available for installation from PyPI:

```
pip install rftokenizer
```

Or you can clone this repo and run

```
python setup.py install
```

## Introduction

This is a simple tokenizer for word-internal segmentation in morphologically rich languages such as Hebrew, Coptic or Arabic, which have big 'super-tokens' (space-delimited words which contain e.g. clitics that need to be segmented) and 'sub-tokens' (the smaller units contained in super-tokens).

Segmentation is based on character-wise binary classification: each character is predicted to have a following border or not. The tokenizer relies on an xgboost classifier, which is fast, very accurate using little training data, and resists overfitting. Solutions do not represent globally optimal segmentations (there is no CRF layer, transition lattices or similar), but at the same time a globally coherent segmentation of each string into known morphological categories is not required, which leads to better OOV item handling. The tokenizer is optimal for medium amounts of data (10K - 200K examples of word forms to segment), and works out of the box with fairly simple dependencies and small model files (see Requirements). For two languages as of summer 2019, RFTokenizer either provides the highest published segmentation accuracy on the official test set (Hebrew) or forms part of an ensemble which does so (Coptic).

To cite this tool, please refer to the following paper:

Zeldes, Amir (2018) A Characterwise Windowed Approach to Hebrew Morphological Segmentation. In: *Proceedings of the 15th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology*. Brussels, Belgium, 101-110.

```
@InProceedings{,
  author    = {Amir Zeldes},
  title     = {A CharacterwiseWindowed Approach to {H}ebrew Morphological Segmentation},
  booktitle = {Proceedings of the 15th {SIGMORPHON} Workshop on Computational Research in Phonetics, Phonology, and Morphology},
  year      = {2018},
  address   = {Brussels, Belgium},
  pages = {101--110}
}
```

The data provided for the Hebrew segmentation experiment in this paper, given in the data/ directory, is derived from the Universal Dependencies version of the Hebrew Treebank, which is made available under a CC BY-NC-SA 4.0 license, but using the earlier splits from the 2014 SPMRL shared task. For attribution information for the Hebrew Treebank, see: https://github.com/UniversalDependencies/UD_Hebrew-HTB . The out-of-domain Wikipedia dataset from the paper, called Wiki5K and available in the data/ directory, is available under the same terms as Wikipedia.

Coptic data is derived from Coptic Scriptorium corpora, see more information at http://www.copticscriptorium.org/

Arabic data is derived from the Prague Arabic Dependency Treebank (UD_Arabic-PADT, https://github.com/UniversalDependencies/UD_Arabic-PADT)

## Performance

Realistic scores on the SPMRL Hebrew dataset (UD_Hebrew, V1 splits), using BERT-based predictions and lexicon data as features, trained jointly on SPMRL and other UD Hebrew IAHLT data:

```
Perfect word forms: 0.9933281004709577
Precision: 0.9923298178331735
Recall: 0.9871244635193133
F-Score: 0.9897202964379631
```

Clean experimental scores on the SPMRL Hebrew dataset (UD_Hebrew, V1 splits), using BERT-based predictions and lexicon data as features and training only on SPMRL:

```
Perfect word forms: 0.9918367346938776
Precision: 0.9885304659498207
Recall: 0.9864091559370529
F-Score: 0.9874686716791979
```

Or the latter without BERT:

```
Perfect word forms: 0.9821036106750393
Precision: 0.9761790182868142
Recall: 0.967103694874851
F-Score: 0.9716201652496708
```

Scores on Hebrew Wiki5K (out-of-domain, with BERT, train on SPMRL):

```
Perfect word forms: 0.9907224634820371
Precision: 0.9851075565361279
Recall: 0.9845644983461963
F-Score: 0.9848359525778881
```

Prague Arabic Dependency Treebank (UD_Arabic-PADT, currently without BERT):

```
Perfect word forms: 0.9846204866724703
Precision: 0.9744331886155331
Recall: 0.9874853343762221
F-Score: 0.9809158451901132
```

Coptic Scriptorium (UD_Coptic-Scriptorium, currently without BERT):

```
Perfect word forms: 0.952007602755999
Precision: 0.9797786292039166
Recall: 0.9637772194304858
F-Score: 0.971712054042643
```

## Requirements

The tokenizer needs:
  * scikit-learn
  * numpy
  * pandas
  * xgboost
  * flair (only if BERT is used)

And if you want to run hyperparameter optimization:
  * hyperopt

Compatible with Python 2 or 3, but compiled models must be specific to Python 2 / 3 (can't use a model trained under Python 2 with Python 3).

## Using

### Command line

To use the tokenizer, include the model file (e.g. `heb.sm3`) in the tokenizer's directory or in `models/`, then select it using `-m heb` and supply a text file to run segmentation on. The input file should have one word-form per line for segmentation.

```
> python tokenize_rf.py -m heb example_in.txt > example_out.txt
```

Input file format:

```
עשרות
אנשים
מגיעים
מתאילנד
לישראל
```

Output format:

```
עשרות
אנשים
מגיעים
מ|תאילנד
ל|ישראל
```

You can also use the option `-n` to separate segments using a newline instead of the pipe character.

### Importing as a module

You can import RFTokenizer once it is installed (e.g. via pip), for example:

```
>>> from rftokenizer import RFTokenizer
>>> my_tokenizer = RFTokenizer(model="ara")
>>> data = open("test_ara.txt",encoding="utf8").read()
>>> tokenized = my_tokenizer.rf_tokenize(data)
>>> print(tokenized)
```

Note that .rf_tokenize() expects a list of word forms to analyze or a string with word forms separated by new lines. The return value is a list of analyses separated by the separator (default: `|`).


## Training

To train a new model, you will need at least a configuration file and a training file. Ideally, you should also provide a lexicon file containing categorized sub-tokens AND super-tokens
and frequency information for sub-tokens (see below).

Training is invoked like this:

```
> python tokenize_rf.py -t -m <LANG> -c <CONF> -l <LEXICON> -f <FREQS> <TRAINING>
```

This will produce `LANG.sm3`, the compiled model (or `.sm2` under Python 2). If `<CONF>` is not supplied, it is assumed to be called `<LANG>.conf`.

If you wish to use BERT features for classification you must *first* train a flair classifier using `flair_pos_tagger.py`, which trains on .conllu data, and name its model `<LANG>.seg`, which should be placed in `models/`. Then train RFTokenizer using the `--bert` option. **Important note**: the data used to train the BERT classifier must be disjoint from the data used to train RFTokenizer, or else it will produce over-reliance (RFTokenizer will learn that BERT is always right, since BERT magically predicts everything correctly, given that it has already seen this training data). Alternatively, you can use a k-fold training regime.

### Configuration

You must specify some settings for your model in a file usually named `LANG.conf`, e.g. heb.conf for Hebrew. This file is a config parser property file with the following format:

  * A section header at the top corresponding to your language model, in brackets, e.g. `[heb]` for Hebrew
  * base_letters - characters to consider during classification. All other characters are treated as `_` (useful for OOV/rare characters, emoji's etc.). These characters should be attested in TRAINING

  Optionally you may add:
  * vowels - if the language distinguishes something like vowels (including matres lectionis), it can be useful to configure them here
  * pos_classes - a mapping of POS tags to collapsed POS tags in the lexicon, in order to reduce sparseness (especially if the tag set is big but training data is small). See below for format.
  * unused - comma separated list of feature names to permanently disable in this model.
  * diacritics - not currently used.
  * regex_tok - a set of regular expressions used for rule based tokenization (e.g. for numbers, see example below)
  * allowed - mapping of characters that may be followed by a boundary at positive positions in the beginning of the word (starting at 0) or negative positions at the end of the word (-1 is the last character). When this setting is used, no other characters/positions will allow splits (useful for languages with a closed vocabulary of affixes). See below for format.

Example `heb.conf` file for Hebrew:

```
[heb]
base_letters=אבגדהוזחטיכלמנסעפצקרשתןםךףץ'-%".?!/,
vowels=אהוי
unused=next_letter
diacritics=ּ
allowed=
	0<-המבלושכ'"-
	1<-המבלשכ'"-
	2<-המבלכ'"-
	3<-ה'"-
	-1<-והיךםן
	-2<-הכנ
regex_tok=
	^([0-9\.,A-Za-z]+)$	\1
	^(ב|ל|מ|כ|ה)([-־])([0-9\./,A-Za-z]+)$	\1|\2|\3
	^(ב|ל|מ|כ|ה)([0-9\./,A-Za-z]+)$	\1|\2
```

If using POS classes:
```
pos_classes=
	V<-VBP|VBZ|VB|VBD|VBG|VBN|MD
	N<-NN|NNP
	NS<-NNS|NNPS
```

### Training file

A two column text file with word forms in one column, and pipe-delimited segmentations in the second column:

```
עשרות	עשרות
אנשים	אנשים
מגיעים	מגיעים
מתאילנד	מ|תאילנד
לישראל	ל|ישראל
...
```

It is assumed that line order is meaningful, i.e. each line provides preceding context for the next line. If you have a **shuffled** corpus of trigrams, you can also supply a four column training file with the columns:

  * Previous wordform
  * Next wordform
  * Current wordform
  * Current wordform segmentation (pipe-separated)

In this case line order is meaningless.

### Lexicon file

The lexicon file is a tab delimited text file with one word form per line and the POS tag assigned to that word in a second column (a third column with lemmas is reserved for future use). Multiple entries per word are possible, e.g.:

```
צלם	NOUN	צלם
צלם	VERB	צילם
צלם	CPLXPP	צל
...
```

It is recommended (but not required) to include entries for complex super-tokens and give them distinct tags, e.g. the sequence צלם above contains two segments: a noun and a possessor clitic. It is therefore given the tag CPLXPP, 'complex including a personal pronoun'. This tag is not used for any simple sub-token segment in the same lexicon.

### Frequency file

The frequency file is a tab delimited text file with one word form per line and the frequency of that word as an integer. Multiple entries per word are possible if pooling data from multiple sources, in which case the sum of integers is taken. In the following example, the frequency of the repeated first item is the sum of the numbers in the first two lines:

```
שמח	32304
שמח	39546
שמט	314
...
```

### Other training options

  * You can specify a train/test split proportion using e.g. `-p 0.2` (default test partition is 0.1 of the data)
  * Variable importances can be outputted using `-i`
  * You can perform retraining on the entire dataset after evaluation of feature importances using `-r`
  * You can ablate certain features using `-a` and a comma separated list of features
  * Hyperparameter optimization can be run with `-o`

If you want to test different classifiers/modify default hyperparameters, you can modify the cross-validation code in the train() routine or use a fixed dev set (look for `cross_val_test`).

