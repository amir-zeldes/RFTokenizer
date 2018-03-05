# RFTokenizer
A character-wise tokenizer for morphologically rich languages

## Introduction

This is a simple tokenizer for word-internal segmentation in languages such as Hebrew, Arabic or Coptic, based on character-wise binary classification: each character is predicted to have a following border or not. The tokenizer relies on scikit-learn ensemble classifiers, which are fast, relatively accurate using little training data, and resist overfitting. However, solutions do not represent globally optimal segmentations (obtainable using a CRF/RNN+CRF or similar). The tokenizer is optimal for medium amounts of data (10K - 100K examples of word forms to segment), and works out of the box with fairly simple dependencies (see Requirements).

## Requirements

The tokenizer needs:
  * scikit-learn (preferably ==0.19.0)
  * numpy
  * pandas
  
Compatible with Python 2 or 3, but compiled models must be specific to Python 2 / 3 (can't use a model trained under Python 2 with Python 3).

## Using

To use the tokenizer, include the model file (e.g. `heb.sm3`) in the tokenizer's directory, then select it using `-m heb` and supply a text file to run segmentation on. The input file should have one word-form per line for segmentation.

```
> python tokenizer_rf.py -m heb some_file.txt > some_file_segmented.txt
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

To train a new model, you will need two files and a configuration statement in `models.conf`. Training is invoked like this:

```
> python tokenize_rf.py -t -m <LANG> -l <LEXICON> <TRAINING>
```

This will produce `LANG.sm3`, the compiled model (or `.sm2` under Python 2).

### Lexicon file

The lexicon file is a tab delimited text file with one word form per line and the POS tag assigned to that word in a second column. Multiple entries per word are possible, e.g.:

```
ⲉ	PREP
ⲉ	PPERO
ⲛⲟⲩⲧⲉ	NOUN
...
```

### Training file

A two column text file with word forms in one column, and pipe-delimited segmentations in the second column:

```
ⲁϥϫⲟⲟⲥ	ⲁ|ϥ|ϫⲟⲟⲥ
ⲇⲉ	ⲇⲉ
ⲛϭⲓⲡⲣⲱⲙⲉ	ⲛϭⲓ|ⲡ|ⲣⲱⲙⲉ
...
```

It is assumed that line order is meaningful, i.e. each line provides preceding context for the next line. If you have a shuffled corpus of trigrams, you can also supply a four column training file with the columns:

  * Previous wordform
  * Next wordform
  * Current wordform
  * Current wordform segmentation (pipe-separated)

In this case line order is meaningless.

### Configuration

You must specify some settings for your model in `models.conf`: 

  * A section corresponding to your language model must be specified in brackets, e.g. `[cop]` for Coptic
  * base_letters - characters to consider during classification. All other characters are treated as `_` (useful for OOV/rare characters, emoji's etc.). These characters should be attested in TRAINING
  * vowels - can be left empty, but if the language distinguishes something like vowels (including matres lectionis), it is useful to configure them here
  * pos_classes - optionally, you may specify a mapping of POS tags to collapsed POS tags in the lexicon, in order to reduce sparseness (especially if the tag set is big but training data is small). See below for format.

Example `models.conf` file for Hebrew and Coptic:
```
[heb]
base_letters=בגדהוזחטיכלמנסעפצקרשתןםךףץ'-%
vowels=אהוי

[cop]
base_letters=ⲁⲃⲅⲇⲉⲍⲏⲑⲓⲕⲗⲙⲛⲝⲟⲡⲣⲥⲧⲩⲫⲭⲯⲱϣϥϩϫϭϯ
vowels=ⲁⲉⲓⲟⲩⲱⲏ
pos_classes=
	A<-AJUS|APST|ACOND|AAOR|ACOND_PPERS|ACONJ|ACONJ_PPERS|AFUTCONJ|ALIM|ANEGAOR|ANEGJUS|ANEGOPT|ANEGOPT_PPERS|ANEGPST|ANEGPST_PPERS|ANY|AOPT|AOPT_PPERS|APREC|EXIST
	I<-ADV|CONJ|COP|FM|NEG|NUM|PINT|PTC|PUNCT|UNKNOWN
	C<-CCIRC|CCIRC_PPERS|CFOC|CPRET|CPRET_PPERS|CREL
	D<-ART|PDEM|PPOS
	X<-ACAUS
	F<-FUT
	P<-IMOD|PREP|PREP_PPERO
	N<-N|NPROP
	O<-PPERS|PPERO|PPERI|IMOD_PPERO|APST_PPERS|CFOC_PPERS|CREL_PPERS
	V<-V|VBD|VSTAT|VIMP|V_PPERO
```

### Options

  * You can specify a train/test split proportion using e.g. `-p 0.2` (default test partition is 0.1 of the data)
  * You can perform retraining on the entire dataset after evaluation using `-r`
  * Variable importances can be outputted using `-i`

If you want to test different classifiers/hyperparameters, there is some cross-validation code in the train() routine (look for `cross_val_test`).