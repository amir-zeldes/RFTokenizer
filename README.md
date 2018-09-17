# RFTokenizer
A character-wise tokenizer for morphologically rich languages

For replication of paper results see replication.md

For full NLP pipelines for morphologically rich languages (MRLs) based on this tool, see:

  * Coptic: http://www.github.com/CopticScriptorium/coptic-nlp/
  * Hebrew: http://www.github.com/amir-zeldes/HebPipe/

## Introduction

This is a simple tokenizer for word-internal segmentation in morphologically rich languages such as Hebrew, Coptic or Arabic, which have big 'super-tokens' (space-delimited words which contain e.g. clitics that need to be segmented) and 'sub-tokens' (the smaller units contained in super-tokens).

Segmentation is based on character-wise binary classification: each character is predicted to have a following border or not. The tokenizer relies on scikit-learn ensemble classifiers, which are fast, relatively accurate using little training data, and resist overfitting. However, solutions do not represent globally optimal segmentations (obtainable using a CRF/RNN+CRF or similar). The tokenizer is optimal for medium amounts of data (10K - 100K examples of word forms to segment), and works out of the box with fairly simple dependencies (see Requirements).

To cite this tool, please refer to the following paper:

Zeldes, Amir (2018) A Characterwise Windowed Approach to Hebrew Morphological Segmentation. In: *Proceedings of the 15th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology*. Brussels, Belgium.

```
@InProceedings{,
  author    = {Amir Zeldes},
  title     = {A CharacterwiseWindowed Approach to {H}ebrew Morphological Segmentation},
  booktitle = {Proceedings of the 15th {SIGMORPHON} Workshop on Computational Research in Phonetics, Phonology, and Morphology},
  year      = {2018},
  address   = {Brussels, Belgium}
}
```

The data provided for the Hebrew segmentation experiment in this paper, given in the data/ directory, is derived from the Universal Dependencies version of the Hebrew Treebank, which is made available under a CC BY-NC-SA 4.0 license, but using the earlier splits from the 2014 SPMRL shared task. For attribution information for the Hebrew Treebank, see: https://github.com/UniversalDependencies/UD_Hebrew-HTB . The out-of-domain Wikipedia dataset from the paper, called Wiki5K and available in the data/ directory, is available under the same terms as Wikipedia.

Coptic data is derived from Coptic Scriptorium corpora, see more information at http://www.copticscriptorium.org/

## Requirements

The tokenizer needs:
  * scikit-learn (preferably ==0.19.0)
  * numpy
  * pandas

Compatible with Python 2 or 3, but compiled models must be specific to Python 2 / 3 (can't use a model trained under Python 2 with Python 3).

## Using

To use the tokenizer, include the model file (e.g. `heb.sm3`) in the tokenizer's directory, then select it using `-m heb` and supply a text file to run segmentation on. The input file should have one word-form per line for segmentation.

```
> python tokenizer_rf.py -m heb example_in.txt > example_out.txt
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

## Training

To train a new model, you will need at least a configuration file and a training file. Ideally, you should also provide a lexicon file containing categorized sub-tokens AND super-tokens
and frequency information for sub-tokens (see below).

Training is invoked like this:

```
> python tokenize_rf.py -t -m <LANG> -c <CONF> -l <LEXICON> -f <FREQS> <TRAINING>
```

This will produce `LANG.sm3`, the compiled model (or `.sm2` under Python 2). If <CONF> is not supplied, it is assumed to be called <LANG>.conf.

### Configuration

You must specify some settings for your model in a file usually named `LANG.conf`, e.g. heb.conf for Hebrew. This file is a config parser property file with the following format:

  * A section header at the top corresponding to your language model, in brackets, e.g. `[heb]` for Hebrew
  * base_letters - characters to consider during classification. All other characters are treated as `_` (useful for OOV/rare characters, emoji's etc.). These characters should be attested in TRAINING

  Optionally you may add:
  * vowels - if the language distinguishes something like vowels (including matres lectionis), it can be useful to configure them here
  * pos_classes - a mapping of POS tags to collapsed POS tags in the lexicon, in order to reduce sparseness (especially if the tag set is big but training data is small). See below for format.
  * unused - comma separated list of feautre names to permanently disable in this model.
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
  * You can ablate certain features using `-a` and a comma separated list of feautres

If you want to test different classifiers/hyperparameters, there is some cross-validation code in the train() routine (look for `cross_val_test`).


