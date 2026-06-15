#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
RFTokenizer - Automatic segmentation of complex word forms
for Morphologically Rich Languages (MRLs)
"""

__version__ = "3.0.0"
__author__ = "Amir Zeldes"
__copyright__ = "Copyright 2018-2026, Amir Zeldes"
__license__ = "Apache 2.0"

import sys, os, random, re, json, zipfile, logging, tempfile
from collections import defaultdict
from pathlib import Path
import configparser

import numpy as np
import xgboost as xgb

logging.disable(logging.INFO)
random.seed(42)

script_dir = os.path.dirname(os.path.realpath(__file__))

CAT_LABELS_DEFAULT = [
    'group_in_lex', 'current_letter', 'prev_prev_letter', 'prev_letter', 'next_letter', 'next_next_letter',
    'mns4_coarse', 'mns3_coarse', 'mns2_coarse', 'mns1_coarse', 'pls1_coarse', 'pls2_coarse',
    'pls3_coarse', 'pls4_coarse', "so_far_pos", "remaining_pos", "prev_grp_pos", "next_grp_pos",
    "remaining_pos_mns1", "remaining_pos_mns2", "prev_grp_first", "prev_grp_last", "next_grp_first", "next_grp_last"
]

NUM_LABELS_DEFAULT = [
    'idx', 'len_bound_group', "current_vowel", "prev_prev_vowel", "prev_vowel", "next_vowel",
    "next_next_vowel", "prev_grp_len", "next_grp_len", "freq_ratio"
]


def read_lex(short_pos, lex_file):
    """
    Read a tab delimited lexicon file. The first two columns must be word-form and POS tag.
    """
    with open(lex_file, 'r', encoding="utf8") as f:
        lex_lines = f.readlines()

    lex = defaultdict(set)
    for line in lex_lines:
        line = line.strip()
        if "\t" in line:
            word, pos = line.split("\t")[0:2]
            if pos in short_pos:
                lex[word].add(short_pos[pos])
            else:
                lex[word].add(pos)

    pos_lookup = {}
    for word in lex:
        pos_lookup[word] = "|".join(sorted(list(lex[word])))

    return pos_lookup


def make_prev_next(seg_table):
    """
    Function to make two column table into a four column table with prev/next seg
    """
    prev_group = "_"
    segs = [tuple(i.split('\t')) for i in seg_table]
    out_segs = []
    for i, line in enumerate(segs):
        current_group, segmentation = line
        next_group = segs[i + 1][0] if i < len(seg_table) - 1 else "_"
        if i > 0:
            prev_group = segs[i - 1][0]
        out_segs.append("\t".join([prev_group, next_group, current_group, segmentation]))
    return out_segs


def segs2array(segs):
    """ Converts piped segments into a binary array of boundary indicators """
    output = []
    cursor = 0
    while cursor < len(segs) - 1:
        if segs[cursor + 1] == "|":
            output.append(1)  # 1 = positive class boundary
            cursor += 2
        else:
            output.append(0)  # 0 = negative class boundary
            cursor += 1
    output.append(0)
    return output


def hyper_optimize(data_x, data_y, val_x=None, val_y=None, space=None, max_evals=20):
    """ Imports training libraries locally to perform hyperparameter optimization """
    from hyperopt import tpe, hp, space_eval, Trials
    from hyperopt.fmin import fmin
    from hyperopt.pyll.base import scope
    from sklearn.metrics import make_scorer, f1_score
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from xgboost import XGBClassifier

    trials = Trials(exp_key="tokenize_tuning")
    average = "micro" if space and "average" in space else "binary"

    def f1_sklearn(truth, predictions):
        return -f1_score(truth, predictions, average=average)

    f1_scorer = make_scorer(f1_sklearn)

    def objective(in_params):
        params = {
            'n_estimators': int(in_params['n_estimators']),
            'max_depth': int(in_params['max_depth']),
            'eta': float(in_params['eta']),
            'gamma': float(in_params['gamma']),
            'colsample_bytree': float(in_params['colsample_bytree']),
            'subsample': float(in_params['subsample'])
        }
        clf = XGBClassifier(random_state=42, n_jobs=4, **params)

        if val_x is None:
            score = cross_val_score(clf, data_x, data_y, scoring=f1_scorer, cv=StratifiedKFold(n_splits=3),
                                    n_jobs=3).mean()
        else:
            clf.fit(data_x, data_y)
            pred_y = clf.predict(val_x)
            score = -f1_score(val_y, pred_y)

        print(f"F1 {-score:.3f} params {params} XGB")
        return score

    if space is None:
        space = {
            'n_estimators': scope.int(hp.quniform('n_estimators', 100, 250, 10)),
            'max_depth': scope.int(hp.quniform('max_depth', 8, 35, 1)),
            'eta': scope.float(hp.quniform('eta', 0.01, 0.2, 0.01)),
            'gamma': scope.float(hp.quniform('gamma', 0.01, 0.2, 0.01)),
            'colsample_bytree': hp.choice('colsample_bytree', [0.6, 0.7, 0.8, 1.0]),
            'subsample': hp.choice('subsample', [0.6, 0.7, 0.8, 0.9, 1.0])
        }

    sys.stderr.write(f"o Using {data_x.shape[0]} tokens to choose hyperparameters\n")
    best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    best_params = space_eval(space, best_params)
    sys.stderr.write(str(best_params) + "\n")

    best_clf = XGBClassifier(random_state=42, n_jobs=4)
    best_clf.set_params(**best_params)
    return best_clf, best_params


class RFTokenizer:
    """ Main tokenizer class used for training and prediction """

    def __init__(self, model):
        self.loaded = False
        self.model = model
        self.lang = Path(model).stem
        self.conf = {}
        self.regex_tok = None
        self.enforce_allowed = False
        self.short_pos = {}
        self.pos_lookup = defaultdict(lambda: "_")
        self.test_cache = {}
        self.allowed = defaultdict(list)

        self.conf["vowels"] = set()
        self.conf["diacritics"] = set()
        self.conf["unused"] = set()

        self.encoders = {}
        self.freqs = {}
        self.booster = None
        self.bert = None
        self.conf_file_parser = None
        self.conf_file_path = None

        # Load local copies of labels that can be ablated
        self.cat_labels = list(CAT_LABELS_DEFAULT)
        self.num_labels = list(NUM_LABELS_DEFAULT)

    def read_conf_file(self, file_name=None, conf_string=None):
        config = configparser.RawConfigParser()

        if conf_string is not None:
            config.read_string(conf_string)
        else:
            if file_name is None:
                file_name = self.model + ".conf"
                sys.stderr.write(f"o Assuming conf file is called {file_name}\n")

            path = Path(file_name) if os.sep in file_name else Path(script_dir) / file_name
            if not path.is_file():
                sys.stderr.write(f"FATAL: could not find configuration file {file_name}\n")
                sys.exit(1)

            config.read(path, encoding="utf8")
            self.conf_file_path = str(path)

        self.conf_file_parser = config

        if self.lang in config:
            for key, val in config.items(self.lang):
                if key in ["base_letters", "vowels", "diacritics", "unused"]:
                    vals = [v.strip() for v in val.split(",")] if key == "unused" else list(val)
                    self.conf[key] = set(vals)
                elif key == "pos_classes":
                    mappings = val.strip().replace("\r", "").split('\n')
                    for mapping in mappings:
                        if "<-" in mapping:
                            target, sources = mapping.strip().split("<-")
                            for source in sources.split("|"):
                                self.short_pos[source] = target
                elif key == "regex_tok":
                    self.regex_tok = []
                    for regex in val.strip().split("\n"):
                        if "\t" in regex:
                            f, r = regex.strip().split("\t")
                            self.regex_tok.append((re.compile(f), r))
                elif key == "allowed":
                    self.enforce_allowed = True
                    for rule in val.strip().split("\n"):
                        if "<-" in rule:
                            position, chars = rule.strip().split("<-")
                            self.allowed[int(position)] = list(chars)

    def load(self, model_path=None):
        """
        Load an .rft Zip archive into memory natively.
        """
        if model_path is None:
            model_path = self.model if self.model.endswith('.rft') else self.model + '.rft'

        search_paths = [
            Path(model_path),
            Path(sys.argv[0]).parent / (self.lang + ".rft"),
            Path(__file__).parent / (self.lang + ".rft"),
            Path(__file__).parent / "models" / (self.lang + ".rft")
        ]

        target_path = next((p for p in search_paths if p.exists()), None)

        if not target_path:
            # Check for legacy pickle fallback attempt to instruct user
            legacy_paths = [p.with_suffix('.sm3') for p in search_paths]
            if any(p.exists() for p in legacy_paths):
                sys.stderr.write(
                    "\nFATAL: Found an old .sm3 Pickle model. This format has been deprecated for security and performance reasons.\n")
                sys.stderr.write(
                    "Please retrain your model to generate the new .rft format using the --train flag.\n\n")
                sys.exit(1)
            raise FileNotFoundError(f"Could not locate model archive for {self.lang}")

        try:
            with zipfile.ZipFile(target_path, 'r') as archive:
                # 1. Load Metadata
                metadata = json.loads(archive.read('metadata.json').decode('utf-8'))
                self.encoders = metadata['encoders']
                self.freqs = metadata['freqs']
                self.cat_labels = metadata['cat_labels']
                self.num_labels = metadata['num_labels']

                self.pos_lookup = defaultdict(lambda: "_")
                self.pos_lookup.update(metadata['pos_lookup'])

                # 2. Load Config
                conf_string = archive.read('config.ini').decode('utf-8')
                self.read_conf_file(conf_string=conf_string)

                # 3. Load XGBoost directly from memory
                self.booster = xgb.Booster()
                self.booster.load_model(bytearray(archive.read('model.json')))

        except zipfile.BadZipFile:
            sys.stderr.write(f"\nFATAL: The file {target_path} is not a valid .rft archive.\n")
            sys.exit(1)

        if "bert" in self.cat_labels:
            try:
                from flair_pos_tagger import FlairTagger
                self.bert = FlairTagger(seg=True)
            except ImportError:
                sys.stderr.write("WARN: Model expects BERT predictions but Flair could not be imported.\n")
                self.bert = None

        self.loaded = True

    @staticmethod
    def words2sents(word_list):
        sents, words = [], []
        for word in word_list:
            words.append(word)
            if word == "." or len(words) > 100:
                sents.append("\n".join(words))
                words = []
        if words:
            sents.append("\n".join(words))
        return sents

    def _get_char_feats(self, word, prev_word, next_word, freqs, pos_lookup, idx, bert_pred=None):
        """ Core internal method to generate features for a single character natively as a dictionary """
        feats = {}

        feats['idx'] = idx
        feats['len_bound_group'] = len(word)
        feats['group_in_lex'] = pos_lookup.get(word, "_")

        feats['current_letter'] = word[idx]
        feats['current_vowel'] = 1 if word[idx] in self.conf['vowels'] else 0

        feats['prev_letter'] = word[idx - 1] if idx > 0 else "_"
        feats['prev_vowel'] = (1 if word[idx - 1] in self.conf['vowels'] else 0) if idx > 0 else -1
        feats['prev_prev_letter'] = word[idx - 2] if idx > 1 else "_"
        feats['prev_prev_vowel'] = (1 if word[idx - 2] in self.conf['vowels'] else 0) if idx > 1 else -1

        feats['next_letter'] = word[idx + 1] if idx < len(word) - 1 else "_"
        feats['next_vowel'] = (1 if word[idx + 1] in self.conf['vowels'] else 0) if idx < len(word) - 1 else -1
        feats['next_next_letter'] = word[idx + 2] if idx < len(word) - 2 else "_"
        feats['next_next_vowel'] = (1 if word[idx + 2] in self.conf['vowels'] else 0) if idx < len(word) - 2 else -1

        so_far_substr = word[:idx + 1]
        remaining_substr = word[idx + 1:]
        remaining_substr_mns1 = word[max(0, idx):]
        remaining_substr_mns2 = word[max(0, idx - 1):]

        feats['so_far_pos'] = pos_lookup.get(so_far_substr, "_")
        feats['remaining_pos'] = pos_lookup.get(remaining_substr, "_")
        feats['remaining_pos_mns1'] = pos_lookup.get(remaining_substr_mns1, "_")
        feats['remaining_pos_mns2'] = pos_lookup.get(remaining_substr_mns2, "_")

        # POS lookup contexts
        for prev_char in [-4, -3, -2, -1, 1, 2, 3, 4]:
            header_prefix = f"mns{abs(prev_char)}_" if prev_char < 0 else f"pls{abs(prev_char)}_"
            substr = ""
            if prev_char < 0 and idx + prev_char >= 0:
                substr = word[idx + prev_char:idx + 1]
            elif prev_char > 0 and idx + prev_char <= len(word):
                substr = word[idx:idx + prev_char + 1]
            feats[header_prefix + "coarse"] = pos_lookup.get(substr, "_")

        # Group Contexts
        feats['prev_grp_first'] = prev_word[0] if prev_word and prev_word != "_" else "_"
        feats['prev_grp_last'] = prev_word[-1] if prev_word and prev_word != "_" else "_"
        feats['prev_grp_len'] = len(prev_word) if prev_word and prev_word != "_" else 0
        feats['prev_grp_pos'] = pos_lookup.get(prev_word, "_") if prev_word and prev_word != "_" else "_"

        feats['next_grp_first'] = next_word[0] if next_word and next_word != "_" else "_"
        feats['next_grp_last'] = next_word[-1] if next_word and next_word != "_" else "_"
        feats['next_grp_len'] = len(next_word) if next_word and next_word != "_" else 0
        feats['next_grp_pos'] = pos_lookup.get(next_word, "_") if next_word and next_word != "_" else "_"

        # Frequencies
        if not freqs:
            feats['freq_ratio'] = 0.0
        else:
            f_sofar = freqs.get(so_far_substr, 0.0)
            f_remain = freqs.get(remaining_substr, 0.0)
            f_whole = freqs.get(word, 0.0) + 1e-10
            feats['freq_ratio'] = (f_sofar * f_remain) / f_whole

        if bert_pred is not None:
            feats['bert'] = bert_pred

        return feats

    def train(self, train_file, lexicon_file=None, freq_file=None, test_prop=0.1, output_importances=False,
              dump_model=False,
              cross_val_test=False, output_errors=False, ablations=None, do_shuffle=True, conf=None, bert=False,
              prune_lex=0.1):

        import pandas as pd
        from sklearn.metrics import accuracy_score
        from xgboost import XGBClassifier

        self.read_conf_file(file_name=conf)
        self.pos_lookup = read_lex(self.short_pos, lexicon_file) if lexicon_file else {}

        if lexicon_file is None:
            print("i WARN: No lexicon file provided, learning purely from examples")

        with open(train_file, 'r', encoding="utf8") as f:
            seg_table = f.read().replace("\r", "").strip().split("\n")

        sys.stderr.write("o Encoding Training data\n")

        non_tab_lines = sum(1 for line in seg_table if "\t" not in line)
        if non_tab_lines > 0:
            sys.stderr.write(f"FATAL: found {non_tab_lines} rows in training data not containing tab\n")
            sys.exit(1)

        if seg_table[0].count("\t") == 1:
            seg_table = make_prev_next(seg_table)
        elif bert:
            sys.stderr.write("WARN: bert sequence predictions may be unreliable if training data is shuffled!\n")

        if bert:
            from flair_pos_tagger import FlairTagger
            lines = [l.split("\t")[2] for l in seg_table if "\t" in l]
            sents = self.words2sents(lines)
            neural_seg = FlairTagger(seg=True)
            bert_preds_raw = neural_seg.predict("\n\n".join(sents), in_format="flair", out_format="xg", as_text=True,
                                                seg=True)
            bert_preds = [line.split("\t")[0] for line in bert_preds_raw.split("\n") if "\t" in line]

            default_rare = "CDO" if self.model == "cop" else "WBB"
            bert_preds = [default_rare if p not in bert_preds else p for p in bert_preds]
            bert_preds = ["O"] + bert_preds
            if "bert" not in self.cat_labels:
                self.cat_labels.append("bert")

        seg_table = ["_\t_\t_\t_"] + seg_table

        freqs = defaultdict(float)
        if freq_file:
            with open(freq_file, 'r', encoding="utf8") as f:
                flines = f.read().replace("\r", "").split("\n")
            total_segs = sum(float(l.split("\t")[1]) for l in flines if l.count("\t") == 1)
            for l in flines:
                if l.count("\t") == 1:
                    w, f_val = l.split("\t")
                    freqs[w] = float(f_val) / total_segs

        self.freqs = freqs
        if not freqs:
            sys.stderr.write("o No segment frequencies provided, adding 'freq_ratio' to ablated features\n")
            ablations = "freq_ratio" if ablations is None else ablations + ",freq_ratio"

        step = int(1 / test_prop) if test_prop > 0 else 0
        test_indices = set(range(0, len(seg_table), step)) if step > 0 else set()

        pruned_pos_lookup = defaultdict(lambda: "_")
        if prune_lex:
            train_freqs = defaultdict(int)
            for row_idx, row in enumerate(seg_table):
                if row_idx not in test_indices:
                    _, _, bound_group, segmentation = row.split("\t")
                    train_freqs[bound_group] += 1
                    for seg in segmentation.split("|"):
                        train_freqs[seg] += 1

            for word, tag in self.pos_lookup.items():
                if train_freqs[word] > 1 or random.random() > prune_lex:
                    pruned_pos_lookup[word] = tag
            sys.stderr.write(f"o Pruning lexicon during training\n")

        all_encoded_rows = []
        words = []
        data_y = []
        bug_rows = []

        for row_idx, row in enumerate(seg_table):
            is_test = 1 if row_idx in test_indices else 0
            prev_group, next_group, bound_group, segmentation = row.split("\t")

            if bound_group != "|" and len(bound_group) != len(segmentation.replace("|", "")):
                bug_rows.append((row_idx, bound_group, segmentation))
                continue

            words.append(bound_group)
            b_pred = bert_preds[row_idx] if bert else None
            if b_pred == "CDO": b_pred = "X"

            y_arr = segs2array(segmentation)
            data_y.extend(y_arr)

            for char_idx in range(len(bound_group)):
                char_feats = self._get_char_feats(bound_group, prev_group, next_group, self.freqs, pruned_pos_lookup,
                                                  char_idx, b_pred)
                char_feats['is_test'] = is_test
                char_feats['grp_id'] = row_idx
                all_encoded_rows.append(char_feats)

        sys.stderr.write(f"o Finished encoding {len(data_y)} chars\n")

        if bug_rows:
            sys.stderr.write(f"i WARN: ignored {len(bug_rows)} unaligned segmentation rows\n")

        # Apply ablations and unused configs
        to_remove = set(self.conf.get("unused", []))
        if ablations and ablations != "none":
            sys.stderr.write("o Ablating features:\n")
            for feat in ablations.split(","):
                sys.stderr.write(f"\t{feat}\n")
                to_remove.add(feat)

        self.cat_labels = [c for c in self.cat_labels if c not in to_remove]
        self.num_labels = [n for n in self.num_labels if n not in to_remove]

        sys.stderr.write("o Creating dataframe and fitting categorical mappings\n")
        df = pd.DataFrame(all_encoded_rows)
        df['boundary'] = data_y

        # Build Categorical Encoders purely in Python dictionaries
        self.encoders = {}
        for col in self.cat_labels:
            if col in df.columns:
                unique_vals = df[col].unique()
                val_to_int = {"_": 0}  # Ensure default mapping is always 0
                idx = 1
                for val in sorted(unique_vals):
                    if val != "_":
                        val_to_int[val] = idx
                        idx += 1
                self.encoders[col] = val_to_int
                df[col] = df[col].map(val_to_int).fillna(0).astype(int)

        strat_train_set = df[df["is_test"] == 0]
        strat_test_set = df[df["is_test"] == 1]

        train_x = strat_train_set[self.cat_labels + self.num_labels].values
        train_y_bin = strat_train_set['boundary'].values

        if self.model == "ara":
            clf = XGBClassifier(n_estimators=170, n_jobs=3, random_state=42, max_depth=24, subsample=1.0,
                                colsample_bytree=0.8, eta=.13, gamma=.15)
        elif bert:
            clf = XGBClassifier(n_estimators=210, n_jobs=3, random_state=42, max_depth=17, subsample=1.0,
                                colsample_bytree=0.6, eta=.02, gamma=.18)
        else:
            clf = XGBClassifier(n_estimators=230, n_jobs=3, random_state=42, max_depth=17, subsample=1.0,
                                colsample_bytree=0.6, eta=.07, gamma=.09)

        if cross_val_test:
            clf, _ = hyper_optimize(train_x, train_y_bin)

        sys.stderr.write("o Learning...\n")
        clf.fit(train_x, train_y_bin)

        if test_prop > 0:
            test_x = strat_test_set[self.cat_labels + self.num_labels].values
            test_y_bin = strat_test_set['boundary'].values
            pred = clf.predict(test_x)

            # Prevent splits on the last char of a bound group artificially
            for i, (_, row) in enumerate(strat_test_set.iterrows()):
                if row["idx"] + 1 == row["len_bound_group"]:
                    pred[i] = 0

            print(f"o Binary clf accuracy:\n\t{accuracy_score(test_y_bin, pred)}")

        if output_importances:
            print("\no Feature importances:")
            for name, imp in sorted(zip(self.cat_labels + self.num_labels, clf.feature_importances_),
                                    key=lambda x: x[1], reverse=True):
                print(f"{name} = {imp}")

        if dump_model:
            output_name = f"{self.lang}.rft"
            with zipfile.ZipFile(output_name, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                    tmp_name = tmp.name
                clf.get_booster().save_model(tmp_name)
                with open(tmp_name, 'rb') as f:
                    archive.writestr('model.json', f.read())
                os.remove(tmp_name)

                metadata = {
                    "cat_labels": self.cat_labels,
                    "num_labels": self.num_labels,
                    "encoders": self.encoders,
                    "freqs": dict(self.freqs),
                    "pos_lookup": dict(self.pos_lookup),
                    "version": __version__
                }
                archive.writestr('metadata.json', json.dumps(metadata))

                if self.conf_file_path:
                    with open(self.conf_file_path, 'r', encoding='utf8') as f:
                        archive.writestr('config.ini', f.read())

            print(f"o Saved model archive to {output_name}")

    def rf_tokenize(self, data, sep="|", indices=None, proba=False):
        if not self.loaded:
            self.load()

        if not isinstance(data, list):
            data = data.strip().split() if "\n" in data else [data]

        if indices is not None and len(indices) == 0:
            return []

        do_not_tok_indices = set()
        word_lengths = []
        encoded_groups = []

        bert_preds = None
        if self.bert is not None:
            sents = self.words2sents(data)
            bert_preds_raw = self.bert.predict("\n\n".join(sents), in_format="flair", out_format="xg", as_text=True,
                                               seg=True)
            bert_preds = [line.split("\t")[0] for line in bert_preds_raw.split("\n") if "\t" in line]

        for i, word in enumerate(data):
            if indices is not None and i not in indices:
                continue

            prev_group = data[i - 1] if i > 0 and len(data[i - 1]) > 0 else "_"
            next_group = data[i + 1] if i < len(data) - 1 and len(data[i + 1]) > 0 else "_"
            word = "_" if not word else word

            if len(word) == 1:
                do_not_tok_indices.add(i)
            elif self.regex_tok:
                for f, r in self.regex_tok:
                    if f.match(word):
                        do_not_tok_indices.add(i)

            bert_pred = bert_preds[i] if bert_preds else None
            if bert_pred == "CDO": bert_pred = "X"

            # Direct mapping from characters to integer arrays
            for char_idx in range(len(word)):
                raw_feats = self._get_char_feats(word, prev_group, next_group, self.freqs, self.pos_lookup, char_idx,
                                                 bert_pred)

                num_row = []
                for col in self.cat_labels:
                    # Encoders default to "_" mapping or 0 if unseen
                    default_val = self.encoders[col].get("_", 0)
                    num_row.append(self.encoders[col].get(raw_feats.get(col, "_"), default_val))

                for col in self.num_labels:
                    num_row.append(raw_feats.get(col, 0))

                encoded_groups.append(num_row)

            word_lengths.append(len(word))

        if not encoded_groups:
            return [], [] if proba else []

        if indices is not None:
            data = [data[i] for i in indices]

        prepped = xgb.DMatrix(np.array(encoded_groups), feature_names=self.cat_labels + self.num_labels)

        if proba:
            probas = self.booster.predict(prepped)
            # Binary logistic returns a single float per row if binary:logistic is used
            if len(probas.shape) == 2: probas = probas[:, 1]
            p = [int(p_val > 0.5) for p_val in probas]
            probs_split = np.split(np.array(probas), np.cumsum(word_lengths)[:-1])
        else:
            p = self.booster.predict(prepped)
            if len(p.shape) == 2: p = p[:, 1]
            p = [int(p_val > 0.5) for p_val in p]

        p_words = np.split(p, np.cumsum(word_lengths)[:-1])

        out_tokenized = []
        out_probas = []

        for word_idx, segmentation in enumerate(p_words):
            tokenized = ""
            out_proba = 1.0
            word = data[word_idx]

            if not word:
                out_tokenized.append("")
                if proba: out_probas.append(1.0)
                continue

            if word_idx in do_not_tok_indices:
                if len(word) > 1 and self.regex_tok:
                    for f, r in self.regex_tok:
                        word = f.sub(r, word)
                out_tokenized.append(word)
                if proba: out_probas.append(1.0)
                continue

            if proba:
                segmentation_probas = probs_split[word_idx]

            for char_idx, bit in enumerate(segmentation):
                tokenized += word[char_idx]

                if proba and segmentation_probas[char_idx] < out_proba:
                    out_proba = segmentation_probas[char_idx]

                if bit == 1 and char_idx < len(word) - 1:
                    if self.enforce_allowed:
                        neg_idx = -1 * (len(word) - char_idx - 1)
                        this_char = word[char_idx]
                        next_char = word[char_idx + 1]

                        seg_allowed = False
                        if char_idx in self.allowed and this_char in self.allowed[char_idx]:
                            seg_allowed = True
                        if neg_idx in self.allowed and next_char in self.allowed[neg_idx]:
                            seg_allowed = True

                        if not seg_allowed:
                            continue
                    tokenized += sep

            out_tokenized.append(tokenized)
            if proba: out_probas.append(out_proba)

        return (out_tokenized, out_probas) if proba else out_tokenized


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentTypeError


    def float_proportion(value):
        f = float(value)
        if not (0.0 <= f <= 1.0):
            raise ArgumentTypeError(f"{value} is an invalid proportion. Must be between 0.0 and 1.0")
        return f


    parser = ArgumentParser()
    parser.add_argument("-n", "--newline", action="store_true", help="insert newline instead of pipe between segments")
    parser.add_argument("-m", "--model", action="store", default="cop", help="language model file path or identifier")
    parser.add_argument("-t", "--train", action="store_true", help="run training")
    parser.add_argument("-l", "--lexicon", action="store", default=None, help="lexicon file to use in training")
    parser.add_argument("-f", "--freqs", action="store", default=None, help="frequency file to use in training")
    parser.add_argument("-c", "--conf", action="store", default=None, help="configuration file to use in training")
    parser.add_argument("-b", "--bert", action="store_true", help="use BERT based classifier as training feature")
    parser.add_argument("-i", "--importances", action="store_true",
                        help="output variable importances during test phrase of training", default=False)
    parser.add_argument("-p", "--proportion", action="store", default=0.1, type=float_proportion,
                        help="Proportion of training data to reserve for testing")
    parser.add_argument("-e", "--errors", action="store_true",
                        help="Whether to output errors during training evaluation")
    parser.add_argument("-r", "--retrain_all", action="store_true",
                        help="re-run training on entire dataset after testing")
    parser.add_argument("-a", "--ablations", action="store", default=None,
                        help="comma separated feature names to ablate")
    parser.add_argument("-o", "--optimize", action="store_true", help="run hyperparameter optimization", default=False)
    parser.add_argument("--prune", action="store", type=float_proportion,
                        help="proportion of hapax legomena to exclude from lexicon during training", default=0.1)
    parser.add_argument("-v", "--version", action="store_true", help="print version number and quit")
    parser.add_argument("file", action="store", help="file to tokenize or train on")

    if "-v" in sys.argv or "--version" in sys.argv:
        print("RFTokenizer V" + __version__)
        sys.exit()

    options = parser.parse_args()
    rf_tok = RFTokenizer(options.model)

    if options.train:
        sys.stderr.write("Training...\n")
        rf_tok.train(
            train_file=options.file, lexicon_file=options.lexicon, dump_model=not options.retrain_all,
            freq_file=options.freqs, output_errors=options.errors, output_importances=options.importances,
            test_prop=options.proportion, ablations=options.ablations, conf=options.conf,
            cross_val_test=options.optimize, bert=options.bert, prune_lex=options.prune
        )
        if options.retrain_all:
            print("\no Retraining on complete data set (no test partition)...")
            rf_tok.train(
                train_file=options.file, lexicon_file=options.lexicon, dump_model=True, output_importances=False,
                freq_file=options.freqs, test_prop=0.0, ablations=options.ablations, conf=options.conf,
                bert=options.bert, prune_lex=options.prune
            )
        sys.exit()

    with open(options.file, 'r', encoding="utf8") as f:
        data = f.read().strip().split("\n")

    sep = "\n" if options.newline else "|"
    output = rf_tok.rf_tokenize(data, sep=sep)
    print("\n".join(output))
