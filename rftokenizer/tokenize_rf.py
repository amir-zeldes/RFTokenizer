#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
RFTokenizer - Automatic segmentation of complex word forms
for Morphologically Rich Languages (MRLs)
"""

__version__ = "1.0.1"
__author__ = "Amir Zeldes"
__copyright__ = "Copyright 2018-2019, Amir Zeldes"
__license__ = "Apache 2.0"


import sys, os, io, random, re
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
try:
	from ConfigParser import RawConfigParser as configparser
except ImportError:
	from configparser import RawConfigParser as configparser

PY3 = sys.version_info[0] == 3
script_dir = os.path.dirname(os.path.realpath(__file__))

cat_labels = ['group_in_lex', 'current_letter', 'prev_prev_letter', 'prev_letter', 'next_letter', 'next_next_letter',
			  'mns4_coarse', 'mns3_coarse', 'mns2_coarse',
			  'mns1_coarse', 'pls1_coarse', 'pls2_coarse',
			  'pls3_coarse', 'pls4_coarse', "so_far_pos", "remaining_pos", "prev_grp_pos", "next_grp_pos",
			  "remaining_pos_mns1", "remaining_pos_mns2",
			  "prev_grp_first", "prev_grp_last", "next_grp_first", "next_grp_last"]

num_labels = ['idx', 'len_bound_group', "current_vowel", "prev_prev_vowel", "prev_vowel", "next_vowel",
			  "next_next_vowel", "prev_grp_len", "next_grp_len", "freq_ratio"]


class FloatProportion(object):
	def __init__(self, start, end):
		self.start = start
		self.end = end
	def __eq__(self, other):
		return self.start <= other <= self.end

class LetterConfig:
	def __init__(self,letters=None, vowels=None, pos_lookup=None):

		if letters is None:
			self.letters = defaultdict(set)
			self.vowels = set()
			self.pos_lookup = defaultdict(lambda: "_")
		else:
			letter_cats = ["current_letter", "prev_prev_letter", "prev_letter", "next_letter", "next_next_letter", "prev_grp_first", "prev_grp_last", "next_grp_first", "next_grp_last"]
			self.letters = defaultdict(set)
			if "group_in_lex" in letters or "current_letter" in letters:  # Letters dictionary already instantiated - we are loading from disk
				self.letters.update(letters)
			else:
				for cat in letter_cats:
					self.letters[cat] = letters
			self.vowels = vowels
			self.pos_lookup = defaultdict(lambda: "_")
			self.pos_lookup.update(pos_lookup)


def multicol_fit_transform(dframe, columns):

	if isinstance(columns, list):
		columns = np.array(columns)
	else:
		columns = columns

	encoder_dict = {}
	# columns are provided, iterate through and get `classes_`
	# ndarray to hold LabelEncoder().classes_ for each
	# column; should match the shape of specified `columns`
	all_classes_ = np.ndarray(shape=columns.shape, dtype=object)
	all_encoders_ = np.ndarray(shape=columns.shape, dtype=object)
	all_labels_ = np.ndarray(shape=columns.shape, dtype=object)
	for idx, column in enumerate(columns):
		# instantiate LabelEncoder
		le = LabelEncoder()
		# fit and transform labels in the column
		dframe.loc[:, column] = le.fit_transform(dframe.loc[:, column].values)
		encoder_dict[column] = le
		# append the `classes_` to our ndarray container
		all_classes_[idx] = (column, np.array(le.classes_.tolist(), dtype=object))
		all_encoders_[idx] = le
		all_labels_[idx] = le

	multicol_dict = {"encoder_dict":encoder_dict, "all_classes_":all_classes_,"all_encoders_":all_encoders_,"columns": columns}
	return dframe, multicol_dict


def hyper_optimize(data_x,data_y,val_x=None,val_y=None,cat_labels=None,space=None,max_evals=20):
	from hyperopt import tpe, hp, space_eval, Trials
	from hyperopt.fmin import fmin
	from hyperopt.pyll.base import scope
	from sklearn.metrics import make_scorer, f1_score
	from sklearn.model_selection import cross_val_score, StratifiedKFold

	trials = Trials(exp_key="tokenize_coptic")

	average="binary"
	if space is not None:
		if "average" in space:
			average = "micro"

	def f1_sklearn(truth,predictions):
		if space is not None:
			if "average" in space:
				return -f1_score(truth,predictions)
		return -f1_score(truth,predictions,average=average)

	f1_scorer = make_scorer(f1_sklearn)

	def objective(in_params):
		clf = in_params['clf']
		if clf == "rf":
			clf = RandomForestClassifier(n_jobs=4,random_state=42)
		elif clf == "gbm":
			clf = GradientBoostingClassifier(random_state=42)
		elif clf == "xgb":
			from xgboost import XGBClassifier
			clf = XGBClassifier(random_state=42,nthread=4)
		elif clf == "cat":
			from catboost import CatBoostClassifier
			clf = CatBoostClassifier(random_state=42)
		else:
			clf = ExtraTreesClassifier(n_jobs=4,random_state=42)

		if clf.__class__.__name__ == "XGBClassifier":
			params = {
				'n_estimators': int(in_params['n_estimators']),
				'max_depth': int(in_params['max_depth']),
				'eta': float(in_params['eta']),
				'gamma': float(in_params['gamma']),
				'colsample_bytree': float(in_params['colsample_bytree']),
				'subsample': float(in_params['subsample'])
			}
			if "average" in in_params:
				params["average"] = in_params["average"]
		else:
			params = {
				'n_estimators': int(in_params['n_estimators']),
				'max_depth': int(in_params['max_depth']),
				'min_samples_split': int(in_params['min_samples_split']),
				'min_samples_leaf': int(in_params['min_samples_leaf']),
				'max_features': in_params['max_features']
			}

		clf.set_params(**params)
		if val_x is None:  # No fixed validation set given, perform cross-valiation on train
			score = cross_val_score(clf, data_x, data_y, scoring=f1_scorer, cv=StratifiedKFold(n_splits=3), n_jobs=3).mean()
		else:  # validated on validation set
			clf.fit(data_x,data_y)
			pred_y = clf.predict(val_x)
			score = -f1_score(val_y,pred_y)
		if "Forest" in clf.__class__.__name__:
			shortname = "RF"
		elif "Cat" in clf.__class__.__name__:
			shortname = "CAT"
		elif "XG" in clf.__class__.__name__:
			shortname = "XGB"
		else:
			shortname = "ET" if "Extra" in clf.__class__.__name__ else "GBM"
		print("F1 {:.3f} params {} {}".format(-score, params, shortname))
		return score

	# For large corpora, consider raising max n_estimators up to 350
	if space is None:
		space = {
			'n_estimators': scope.int(hp.quniform('n_estimators', 75, 250, 10)),
			'max_depth': scope.int(hp.quniform('max_depth', 5, 40, 1)),
			'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
			'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
			'max_features': hp.choice('max_features', ["sqrt", None, 0.5, 0.6, 0.7, 0.8]),
			'clf': hp.choice('clf', ["rf","et","gbm"])
		}

	sys.stderr.write("o Using "+str(data_x.shape[0])+" tokens to choose hyperparameters\n")
	if val_x is not None:
		sys.stderr.write("o Using "+str(val_x.shape[0])+" held out tokens as fixed validation data\n")
	else:
		sys.stderr.write("o No validation data provided, using cross-validation on train set to score\n")

	best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

	best_params = space_eval(space,best_params)
	sys.stderr.write(str(best_params) + "\n")

	best_clf = best_params['clf']
	if best_clf == "rf":
		best_clf = RandomForestClassifier(n_jobs=4,random_state=42)
	elif best_clf == "gbm":
		best_clf = GradientBoostingClassifier(random_state=42)
	elif best_clf == "xgb":
		from xgboost import XGBClassifier
		best_clf = XGBClassifier(random_state=42,nthread=4)
	else:
		best_clf = ExtraTreesClassifier(n_jobs=4,random_state=42)
	del best_params['clf']
	best_clf.set_params(**best_params)

	return best_clf, best_params


def multicol_transform(dframe, columns, all_encoders_):
	for idx, column in enumerate(columns):
		dframe.loc[:, column] = all_encoders_[idx].transform(dframe.loc[:, column].values)
	return dframe


#@profile
def bg2array(bound_group, prev_group="", next_group="", print_headers=False, grp_id=-1, is_test=-1, config=None, train=False, freqs=None):

	output = []

	letters = config.letters
	vowels = config.vowels
	pos_lookup = config.pos_lookup

	group_in_lex = pos_lookup[bound_group] if pos_lookup[bound_group] in letters["group_in_lex"] or train else "_"

	for idx in range(len(bound_group)):

		char_feats = []
		so_far_substr = bound_group[:idx+1]
		remaining_substr = bound_group[idx+1:]
		remaining_substr_mns1 = bound_group[max(0,idx):]
		remaining_substr_mns2 = bound_group[max(0,idx-1):]
		so_far_pos = pos_lookup[so_far_substr] if pos_lookup[so_far_substr] in letters["so_far_pos"] or train else "_"
		remaining_pos = pos_lookup[remaining_substr] if pos_lookup[remaining_substr] in letters["remaining_pos"] or train else "_"
		remaining_pos_mns1 = pos_lookup[remaining_substr_mns1] if pos_lookup[remaining_substr_mns1] in letters["remaining_pos_mns1"] or train else "_"
		remaining_pos_mns2 = pos_lookup[remaining_substr_mns2] if pos_lookup[remaining_substr_mns2] in letters["remaining_pos_mns2"] or train else "_"

		current_letter = bound_group[idx] if bound_group[idx] in letters["current_letter"] or train else "_"
		current_vowel = 1 if bound_group[idx] in vowels else 0
		if idx > 0:
			prev_letter = bound_group[idx-1] if bound_group[idx-1] in letters["prev_letter"] or train else "_"
			prev_vowel = 1 if bound_group[idx-1] in vowels else 0
		else:
			prev_letter = "_"
			prev_vowel = -1
		if idx > 1:
			prev_prev_letter = bound_group[idx-2] if bound_group[idx-2] in letters["prev_prev_letter"] else"_"
			prev_prev_vowel = 1 if bound_group[idx-2] in vowels else 0
		else:
			prev_prev_letter = "_"
			prev_prev_vowel = -1
		if idx < len(bound_group)-1:
			next_letter = bound_group[idx+1] if bound_group[idx+1] in letters["next_letter"] else"_"
			next_vowel = 1 if bound_group[idx+1] in vowels else 0
		else:
			next_letter = "_"
			next_vowel = -1
		if idx < len(bound_group)-2:
			next_next_letter = bound_group[idx+2] if bound_group[idx+2] in letters["next_next_letter"] or train else "_"
			next_next_vowel = 1 if bound_group[idx+2] in vowels else 0
		else:
			next_next_letter = "_"
			next_next_vowel = -1

		char_feats += [idx, group_in_lex, current_letter, prev_prev_letter, prev_letter, next_letter, next_next_letter, len(bound_group)]
		char_feats += [current_vowel, prev_prev_vowel, prev_vowel, next_vowel, next_next_vowel, so_far_pos, remaining_pos, remaining_pos_mns1, remaining_pos_mns2]
		headers = ["idx","group_in_lex","current_letter","prev_prev_letter","prev_letter","next_letter","next_next_letter","len_bound_group"]
		headers += ["current_vowel","prev_prev_vowel","prev_vowel","next_vowel","next_next_vowel", "so_far_pos", "remaining_pos","remaining_pos_mns1","remaining_pos_mns2"]

		# POS lookup features
		all_pos_feats = []
		for chargram_idx, prev_char in enumerate([-4,-3,-2,-1,1,2,3,4]):
			substr = ""
			header_prefix = "_"
			if prev_char < 0:
				header_prefix = "mns" + str(abs(prev_char)) + "_"
				if idx + prev_char >= 0:
					substr = bound_group[idx+prev_char:idx+1]
			elif prev_char > 0:
				header_prefix = "pls" + str(abs(prev_char)) + "_"
				if idx + prev_char <= len(bound_group):
					substr = bound_group[idx:idx+prev_char+1]

			coarse_tag = pos_lookup[substr] if pos_lookup[substr] in letters[header_prefix + "coarse"] or train else "_"

			pos_feats = [coarse_tag]
			if print_headers:
				headers.append(header_prefix + "coarse")
			all_pos_feats += pos_feats #clean_pos_feats

		char_feats += all_pos_feats

		if prev_group != "":
			prev_first = prev_group[0] if prev_group[0] in letters["prev_grp_first"] or train else "_"
			prev_last = prev_group[-1] if prev_group[-1] in letters["prev_grp_last"] or train else "_"
			prev_grp_pos = pos_lookup[prev_group] if pos_lookup[prev_group] in letters["prev_grp_pos"] or train else "_"
			prev_feats = [prev_first,prev_last,len(prev_group),prev_grp_pos]
			char_feats += prev_feats
			if print_headers:
				headers += ["prev_grp_first","prev_grp_last","prev_grp_len","prev_grp_pos"]
		if next_group != "":
			next_first = next_group[0] if next_group[0] in letters["next_grp_first"] or train else "_"
			next_last = next_group[-1] if next_group[-1] in letters["next_grp_last"] or train else "_"
			next_grp_pos = pos_lookup[next_group] if pos_lookup[next_group] in letters["next_grp_pos"] or train else "_"
			next_feats = [next_first,next_last,len(next_group),next_grp_pos]
			char_feats += next_feats
			if print_headers:
				headers += ["next_grp_first","next_grp_last","next_grp_len","next_grp_pos"]

		headers += ["freq_ratio"]
		if freqs is None:
			char_feats += [0.0]
		else:
			f_sofar = freqs[so_far_substr]
			f_remain = freqs[remaining_substr]
			f_whole = freqs[bound_group] + 0.0000000001  # Delta smooth whole
			char_feats += [f_sofar*f_remain/f_whole]

		if grp_id > -1:
			char_feats += [grp_id]
			if print_headers:
				headers += ["grp_id"]
		if is_test > -1:
			char_feats += [is_test]
			if print_headers:
				headers += ["is_test"]

		char_feats += [bound_group,prev_group,next_group]
		headers += ["this_group","prev_group","next_group"]

		output.append(char_feats)

	if print_headers:
		return headers
	else:
		return output


def segs2array(segs):
	output = []
	cursor = 0
	while cursor < len(segs) - 1:  # Word not over yet
		if segs[cursor + 1] == "|":
			output.append(1)  # 1 = positive class
			cursor += 2
		else:
			output.append(0)  # 0 = negative class
			cursor += 1
	output.append(0)

	return output


def read_lex(short_pos, lex_file):
	"""
	Read a tab delimited lexicon file. The first two columns must be word-form and POS tag.

	:param short_pos: Dictionary possibly mapping POS tags in lexicon to alternative (usually collapsed) POS tags.
	:param lex_file: Name of file to read.
	:return: defaultdict returning the concatenated POS tags of all possible analyses of each form, or "_" if unknown.
	"""
	lex_lines = io.open(lex_file, encoding="utf8").readlines()
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
	:param seg_table: Input table of form:

			They	They
			don't	do|n't
			know	know

	:return: Four column table with prev/next group context columns:

			_       don't   They    They
			They    know    don't   do|n't
			don't	_       know	know
	"""

	prev_group = "_"
	segs = [tuple(i.split('\t')) for i in seg_table]
	out_segs = []
	for i, line in enumerate(segs):
		current_group, segmentation = line
		if i < len(seg_table) - 1:
			next_group = segs[i+1][0]
		else:
			next_group = "_"  # Last group in data
		if i > 0:
			prev_group = segs[i-1][0]
		out_segs.append("\t".join([prev_group, next_group, current_group, segmentation]))

	return out_segs


class RFTokenizer:
	"""
	Main tokenizer class used for both training and prediction
	"""

	def __init__(self, model):
		"""

		:param model: name of the model, usually a lowercase ISO 3 letter language code, e.g. 'heb' for Hebrew
		"""
		self.loaded = False
		self.model = model
		self.lang = os.path.basename(model).replace(".sm2","").replace(".sm3","")
		self.conf = {}
		self.regex_tok = None
		self.enforce_allowed = False
		self.short_pos = {}
		self.pos_lookup = {}
		self.test_cache = {}  # Cache for bg2array encoded bound groups at test time
		self.allowed = defaultdict(list)
		self.conf["base_letters"] = set()
		self.conf["vowels"] = set()
		self.conf["diacritics"] = set()
		self.conf["unused"] = set()  # For features not used in this model - to temporarily disable features, use option -a (ablations)
		self.conf_file_parser = None

	def read_conf_file(self, file_name=None):
		if self.conf_file_parser is None:
			config = configparser()
			if file_name is None:
				file_name = self.model + ".conf"
				sys.stderr.write("o Assuming conf file is called " + file_name + "\n")
			if os.sep in file_name:  # Path specified
				try:
					config.read_file(io.open(file_name, encoding="utf8"))
				except AttributeError:
					config.readfp(io.open(file_name, encoding="utf8"))
			else:
				if not os.path.isfile(script_dir + os.sep + file_name):
					sys.stderr.write("FATAL: could not find configuration file " + file_name + " in " + script_dir + "\n")
					sys.exit()
				try:
					config.read_file(io.open(script_dir + os.sep + file_name,encoding="utf8"))
				except AttributeError:
					config.readfp(io.open(script_dir + os.sep + file_name,encoding="utf8"))
			self.conf_file_parser = config
		else:
			config = self.conf_file_parser

		for key, val in config.items(self.lang):
			if key in ["base_letters","vowels","diacritics","unused"]:
				if key == "unused":
					vals = val.split(",")
					vals = [v.strip() for v in vals]
				else:
					vals = list(val)
				self.conf[key] = set(vals)
			elif key == "pos_classes":
				mappings = val.strip().replace("\r","").split('\n')
				for mapping in mappings:
					if "<-" in mapping:
						target, sources = mapping.strip().split("<-")
						for source in sources.split("|"):
							self.short_pos[source] = target
			elif key == "regex_tok":
				self.regex_tok = []
				items = val.strip().split("\n")
				for regex in items:
					if "\t" in regex:
						f, r = regex.strip().split("\t")
						self.regex_tok.append((re.compile(f),r))
					else:
						sys.stderr.write("WARN: regex entry without tab in conf file\n")
			elif key == "allowed":
				self.enforce_allowed = True
				items = val.strip().split("\n")
				for rule in items:
					if "<-" in rule:
						position, chars = rule.strip().split("<-")
						try:
							position = int(position)
						except Exception as e:
							raise ValueError("Can't interpret position instruction in conf file as integer: " + position + "\n")
						self.allowed[position] = list(chars)
					else:
						sys.stderr.write("WARN: allowed segmentation position entry without '<-' in conf file\n")
		self.letters = self.conf["base_letters"]

	def load(self, model_path=None):
		"""
		Load a picked model.

		:param model_path: Path to the model pickle file. If not specified, looks for model language name +.sm2 (Python 2) or .sm3 (Python 3), e.g. heb.sm3
		:return: None
		"""

		if model_path is None:
			# Default model path for a language is the language name, extension ".sm2" for Python 2 or ".sm3" for Python 3
			model_path = self.lang + ".sm" + str(sys.version_info[0])
		if not os.path.exists(model_path):  # Try loading from calling directory
			model_path = os.path.dirname(sys.argv[0]) + self.lang + ".sm" + str(sys.version_info[0])
		if not os.path.exists(model_path):  # Try loading from tokenize_rf.py directory
			model_path = os.path.dirname(os.path.realpath(__file__)) + os.sep + self.lang + ".sm" + str(sys.version_info[0])
		if not os.path.exists(model_path):  # Try loading from models/ directory
			model_path = os.path.dirname(os.path.realpath(__file__)) + os.sep + "models" + os.sep + self.lang + ".sm" + str(sys.version_info[0])
		#sys.stderr.write("Module: " + self.__module__ + "\n")
		self.tokenizer, self.num_labels, self.cat_labels, self.multicol_dict, pos_lookup, self.freqs, self.conf_file_parser = joblib.load(model_path)
		default_pos_lookup = defaultdict(lambda :"_")
		default_pos_lookup.update(pos_lookup)
		self.pos_lookup = default_pos_lookup
		self.read_conf_file()
		self.loaded = True

	def train(self, train_file, lexicon_file=None, freq_file=None, test_prop=0.1, output_importances=False, dump_model=False,
			  cross_val_test=False, output_errors=False, ablations=None, dump_transformed_data=False, do_shuffle=True, conf=None):
		"""

		:param train_file: File with segmentations to train on in one of the two formats described in make_prev_next()
		:param lexicon_file: Tab delimited lexicon file with full forms in first column and POS tag in second column (multiple rows per form possible)
		:param freq_file: Tab delimited file with segment forms and their frequencies as integers in two columns
		:param test_prop: (0.0 -- 0.99..) Proportion of shuffled data to test on
		:param output_importances: Whether to print feature importances (only if test proportion > 0.0)
		:param dump_model: Whether to dump trained model to disk via joblib
		:param cross_val_test: Whether to perform cross-validation for hyper parameter optimization
		:param output_errors: Whether to output prediction errors to a file 'errs.txt'
		:param ablations: Comma separated string of feature names to ablate, e.g. "freq_ratio,prev_grp_pos,next_grp_pos"
		:param dump_transformed_data: If true, transform data to a pandas dataframe and write to disk, then quit
				(useful to train other approaches on the same features, e.g. a neural classifier)
		:param do_shuffle: Whether training data is shuffled after context extraction but before test partition is created
				(this has no effect if training on whole training corpus)
		:param conf: configuration file for training (by default: <MODELNAME>.conf)
		:return: None
		"""

		import timing

		self.read_conf_file(file_name=conf)
		pos_lookup = read_lex(self.short_pos,lexicon_file)
		self.pos_lookup = pos_lookup
		conf_file_parser = self.conf_file_parser
		letter_config = LetterConfig(self.letters, self.conf["vowels"], self.pos_lookup)

		np.random.seed(42)

		if lexicon_file is None:
			print("i WARN: No lexicon file provided, learning purely from examples")

		seg_table = io.open(train_file,encoding="utf8").read()
		seg_table = seg_table.replace("\r","").strip()
		for c in self.conf["diacritics"]:  # TODO: configurable diacritic removal
			pass
			#seg_table = seg_table.replace(c,"")
		seg_table = seg_table.split("\n")

		sys.stderr.write("o Encoding Training data\n")

		# Validate training data
		non_tab_lines = 0
		non_tab_row = 0
		for r, line in enumerate(seg_table):
			if line.count("\t") < 1:
				non_tab_lines += 1
				non_tab_row = r
		if non_tab_lines > 0:
			sys.stderr.write("FATAL: found " + str(non_tab_lines) + " rows in training data not containing tab\n")
			sys.stderr.write("       Last occurrence at line: " + str(non_tab_row) + "\n")
			sys.exit()

		# Make into four cols: prev \t next \t current \t segmented (unless already receiving such a table, for shuffled datasets)
		if seg_table[0].count("\t") == 1:
			seg_table = make_prev_next(seg_table)

		# Ensure OOV symbol is in data
		seg_table = ["_\t_\t_\t_"] + seg_table

		data_y = []
		words = []
		all_encoded_groups = []

		encoding_cache = {}
		non_ident_segs = 0

		shuffle_mapping = list(range(len(seg_table)))
		zipped = list(zip(seg_table, shuffle_mapping))

		# Shuffle table to sample across entire dataset if desired
		if do_shuffle and False:
			random.Random(24).shuffle(zipped)

		seg_table, shuffle_mapping = zip(*zipped)

		headers = bg2array("_________",prev_group="_",next_group="_",print_headers=True,is_test=1,grp_id=1,config=letter_config)

		word_idx = -1
		bug_rows = []

		freqs = defaultdict(float)
		total_segs = 0.0
		flines = io.open(freq_file,encoding="utf8").read().replace("\r","").split("\n") if freq_file is not None else []
		for l in flines:
			if l.count("\t")==1:
				w, f = l.split("\t")
				freqs[w] += float(f)
				total_segs += float(f)

		for u in freqs:
			freqs[u] = freqs[u]/total_segs

		# Don't use freqs if they're empty
		if len(freqs) == 0:
			sys.stderr.write("o No segment frequencies provided, adding 'freq_ratio' to ablated features\n")
			if ablations is None:
				ablations = "freq_ratio"
			else:
				if "freq_ratio" not in ablations:
					ablations += ",freq_ratio"

		step = int(1/test_prop) if test_prop > 0 else 0
		test_indices = list(range(len(seg_table)))[0::step] if step > 0 else []
		test_rows = []

		for row_idx, row in enumerate(seg_table):
			is_test = 1 if row_idx in test_indices else 0

			prev_group, next_group, bound_group, segmentation = row.split("\t")
			if bound_group != "|":
				if len(bound_group) != len(segmentation.replace("|","")):  # Ignore segmentations that also normalize
					non_ident_segs += 1
					bug_rows.append((row_idx,bound_group,segmentation.replace("|","")))
					continue

			###
			if dump_transformed_data:
				if is_test:
					test_rows.append(bound_group + "\t" + segmentation)
			###

			word_idx += 1
			words.append(bound_group)
			group_type = "_".join([x for x in [prev_group, next_group, bound_group] if x != ""])
			if group_type in encoding_cache:  # No need to encode, an identical featured group has already been seen
				encoded_group = encoding_cache[group_type]
				for c in encoded_group:
					c[headers.index("is_test")] = is_test  # Make sure that this group's test index is correctly assigned
			else:
				encoded_group = bg2array(bound_group,prev_group=prev_group,next_group=next_group,is_test=is_test,grp_id=word_idx,config=letter_config,train=True,freqs=freqs)
				encoding_cache[group_type] = encoded_group
			all_encoded_groups += encoded_group
			data_y += segs2array(segmentation)

		sys.stderr.write("o Finished encoding " + str(len(data_y)) + " chars (" + str(len(seg_table)) + " groups, " + str(len(encoding_cache)) + " group types)\n")

		if non_ident_segs > 0:
			with io.open("bug_rows.txt",'w',encoding="utf8") as f:
				f.write(("\n".join([str(r) + ": " + g + "<>" + s for r, g, s in sorted([[shuffle_mapping[x], g, s] for x, g, s in bug_rows])]) + "\n"))

			sys.stderr.write("i WARN: found " + str(non_ident_segs) + " rows in training data where left column characters not identical to right column characters\n")
			sys.stderr.write("        Row numbers dumped to: bug_rows.txt\n")
			sys.stderr.write("        " + str(non_ident_segs) + " rows were ignored in training\n\n")

		data_y = np.array(data_y)


		# Remove features switched off in .conf file
		for label in self.conf["unused"]:
			if label in cat_labels:
				cat_labels.remove(label)
			if label in num_labels:
				num_labels.remove(label)

		# Handle temporary ablations if specified in option -a
		if ablations is not None:
			sys.stderr.write("o Applying ablations\n")
			if len(ablations) > 0 and ablations != "none":
				abl_feats = ablations.split(",")
				sys.stderr.write("o Ablating features:\n")
				for feat in abl_feats:
					found = False
					if feat in cat_labels:
						cat_labels.remove(feat)
						found = True
					elif feat in num_labels:
						num_labels.remove(feat)
						found = True
					if found:
						sys.stderr.write("\t"+feat+"\n")
					else:
						sys.stderr.write("\tERR: can't find ablation feature " + feat + "\n")
						sys.exit()

		sys.stderr.write("o Creating dataframe\n")
		data_x = pd.DataFrame(all_encoded_groups, columns=headers)

		###
		if dump_transformed_data:
			data_x["resp"] = data_y
			import csv
			to_remove = ["is_test","grp_id"]  # Columns to remove from transformed data dump
			out_cols = [col for col in headers if col not in to_remove] + ["resp"]  # Add the response column as 'resp'
			data_x.iloc[data_x.index[data_x["is_test"] == 0]].to_csv("rftokenizer_train_featurized.tab",sep="\t",quotechar="",quoting=csv.QUOTE_NONE,encoding="utf8",index=False,columns=out_cols)
			data_x.iloc[data_x.index[data_x["is_test"] == 1]].to_csv("rftokenizer_test_featurized.tab",sep="\t",quotechar="",quoting=csv.QUOTE_NONE,encoding="utf8",index=False,columns=out_cols)
			# Dump raw test rows to compare gold solution
			with io.open("rftokenizer_test_gold.tab","w",encoding="utf8") as gold:
				gold.write("\n".join(test_rows) + "\n")
			sys.stderr.write("o Wrote featurized train/test set and gold test to rftokenizer_*.tab\n")
			sys.exit()
		###

		data_x_enc, multicol_dict = multicol_fit_transform(data_x, pd.Index(cat_labels))

		if test_prop > 0:
			sys.stderr.write("o Generating train/test split with test proportion "+str(test_prop)+"\n")

		data_x_enc["boundary"] = data_y
		strat_train_set = data_x_enc.iloc[data_x_enc.index[data_x_enc["is_test"] == 0]]
		strat_test_set = data_x_enc.iloc[data_x_enc.index[data_x_enc["is_test"] == 1]]

		sys.stderr.write("o Transforming data to numerical array\n")
		train_x = strat_train_set[cat_labels+num_labels].values

		train_y = strat_train_set["boundary"]
		train_y_bin = np.where(strat_train_set['boundary'] == 0, 0, 1)

		if test_prop > 0:
			test_x = strat_test_set[cat_labels+num_labels].values
			test_y_bin = np.where(strat_test_set['boundary'] == 0, 0, 1)
			bound_grp_idx = np.array(strat_test_set['grp_id'])

			from sklearn.dummy import DummyClassifier
			d = DummyClassifier(strategy="most_frequent")
			d.fit(train_x,train_y_bin)
			pred = d.predict(test_x)
			print("o Majority baseline:")
			print("\t" + str(accuracy_score(test_y_bin, pred)))

		# Classifier used in 2018 paper:
		#clf = ExtraTreesClassifier(n_estimators=250, max_features=None, n_jobs=3, random_state=42)

		# Use xgboost for slightly better accuracy than paper
		from xgboost import XGBClassifier

		# Good parameters for Arabic
		if self.model == "ara":
			clf = XGBClassifier(n_estimators=170,n_jobs=3,random_state=42,max_depth=24,subsample=1.0,colsample_bytree=0.8,eta=.13,gamma=.15)
		else:
			# Good parameters for Hebrew/Coptic
			clf = XGBClassifier(n_estimators=230,n_jobs=3,random_state=42,max_depth=17,subsample=1.0,colsample_bytree=0.6,eta=.07,gamma=.09)

		if cross_val_test:
			# Modify code to tune hyperparameters

			from hyperopt import hp
			from hyperopt.pyll import scope
			space = {
				'n_estimators': scope.int(hp.quniform('n_estimators', 100, 250, 10)),
				'max_depth': scope.int(hp.quniform('max_depth', 8, 35, 1)),
				'eta': scope.float(hp.quniform('eta', 0.01, 0.2, 0.01)),
				'gamma': scope.float(hp.quniform('gamma', 0.01, 0.2, 0.01)),
				'colsample_bytree': hp.choice('colsample_bytree', [0.6,0.7,0.8,1.0]),
				'subsample': hp.choice('subsample', [0.6,0.7,0.8,0.9,1.0]),
				'clf': hp.choice('clf', ["xgb"])
			}
			if test_prop > 0:
				best_clf, best_params = hyper_optimize(train_x,train_y_bin,val_x=test_x,val_y=test_y_bin,space=space,max_evals=100)
			else:
				best_clf, best_params = hyper_optimize(train_x,train_y_bin,val_x=None,val_y=None,space=space,max_evals=20)
			print(best_params)
			clf = best_clf

			print("\nBest parameters:\n" + 30 * "=")
			print(best_params)

		sys.stderr.write("o Learning...\n")
		clf.fit(train_x, train_y_bin)

		if test_prop > 0:
			pred = clf.predict(test_x)
			j=-1
			for i, row in strat_test_set.iterrows():
				j+=1
				if row["idx"] +1 == row["len_bound_group"]:
					pred[j] = 0

			print("o Binary clf accuracy:")
			print("\t" + str(accuracy_score(test_y_bin, pred)))

			group_results = defaultdict(lambda : 1)
			for i in range(len(pred)):
				grp = bound_grp_idx[i]
				if test_y_bin[i] != pred[i]:
					group_results[grp] = 0

			correct = 0
			total = 0
			for grp in set(bound_grp_idx):
				if group_results[grp] == 1:
					correct +=1
				total +=1
			print("o Perfect bound group accuracy:")
			print("\t" + str(float(correct)/total))

			errs = defaultdict(int)
			for i, word in enumerate(words):
				if i in group_results:
					if group_results[i] == 0:
						errs[word] += 1

			if output_errors:
				print("o Writing prediction errors to errs.txt")
				with io.open("errs.txt",'w',encoding="utf8") as f:
					for err in errs:
						f.write(err + "\t" + str(errs[err])+"\n")
		else:
			print("o Test proportion is 0%, skipping evaluation")

		if output_importances:
			feature_names = cat_labels + num_labels

			zipped = zip(feature_names, clf.feature_importances_)
			sorted_zip = sorted(zipped, key=lambda x: x[1], reverse=True)
			print("o Feature importances:\n")
			for name, importance in sorted_zip:
				print(name, "=", importance)


		if dump_model:
			plain_dict_pos_lookup = {}
			plain_dict_pos_lookup.update(pos_lookup)
			joblib.dump((clf, num_labels, cat_labels, multicol_dict, plain_dict_pos_lookup, freqs, conf_file_parser), self.lang + ".sm" + str(sys.version_info[0]), compress=3)
			print("o Dumped trained model to " + self.lang + ".sm" + str(sys.version_info[0]))

	def rf_tokenize(self, data, sep="|", indices=None, proba=False):
		"""
		Main tokenizer routine

		:param data: ordered list of word forms (prev/next word context is taken from list, so meaningful order is assumed)
		:param sep: separator to use for found segments, default: |
		:param indices: options; list of integer indices to process. If supplied, positions not in the list are skipped
		:param proba: boolean, return probabilities
		:return: list of word form strings tokenized using the separator
		"""

		if not self.loaded:
			self.load()

		if not isinstance(data,list):
			if "\n" in data:
				data = data.strip().split()
			else:
				data = [data]

		tokenizer, num_labels, cat_labels, multicol_dict, freqs = self.tokenizer, self.num_labels, self.cat_labels, self.multicol_dict, self.freqs

		do_not_tok_indices = set()

		if indices is not None:
			if len(indices) == 0:
				return []

		encoded_groups = []

		headers = bg2array("_________",prev_group="_",next_group="_",print_headers=True,config=LetterConfig())
		word_lengths = []
		cursor = 0

		letters = {}
		for header in headers:
			if header in multicol_dict["encoder_dict"]:
				letters[header] = multicol_dict["encoder_dict"][header].classes_

		letter_config = LetterConfig(letters, self.conf["vowels"], self.pos_lookup)

		j = 0
		for i, word in enumerate(data):
			if indices is not None:
				if i not in indices:
					continue
			prev_group = data[i-1] if i > 0 else "_"
			next_group = data[i+1] if i < len(data)-1 else "_"

			# Protect again zero length input
			if len(prev_group) == 0:
				prev_group = "_"
			if len(next_group) == 0:
				next_group = "_"
			if len(word) == 0:
				word = "_"

			if self.regex_tok is not None:
				for f, r in self.regex_tok:
					if f.match(word) is not None:
						do_not_tok_indices.add(j)
			j += 1

			group_type = "_".join([prev_group, next_group, word])
			if group_type in self.test_cache:  # No need to encode, an identical featured group has already been seen
				encoded_group = self.test_cache[group_type]
			else:
				encoded_group = bg2array(word,prev_group=prev_group,next_group=next_group,config=letter_config,freqs=freqs)
				self.test_cache[group_type] = encoded_group
			encoded_groups += encoded_group
			word_lengths.append(cursor + len(word))
			cursor += len(word)
		word_lengths.pop()

		if indices is not None:
			data = [data[i] for i in indices]

		data_x = pd.DataFrame(encoded_groups)
		data_x.columns = headers

		data_x = multicol_transform(data_x,multicol_dict["columns"],multicol_dict["all_encoders_"])
		prepped = data_x[cat_labels+num_labels].values

		if proba:
			probas = tokenizer.predict_proba(prepped)
			p = [int(p[1]>0.5) for p in probas]
			probs = np.split(np.array([p[1] for p in probas]), word_lengths)
		else:
			p = tokenizer.predict(prepped)
		p_words = np.split(p, word_lengths)
		out_tokenized = []
		out_probas = []
		out_proba = 0.0

		for word_idx, segmentation in enumerate(p_words):
			tokenized = ""
			if word_idx == 90:
				a=5
			if data[word_idx] == "":
				tokenized = ""
			else:
				if word_idx in do_not_tok_indices:
					word = data[word_idx]
					for f, r in self.regex_tok:
						word = f.sub(r, word)
					tokenized += word
					if proba:
						out_proba = 1.0
				else:
					out_proba = 1.0
					if proba:
						segmentation_probas = probs[word_idx]
					for idx, bit in enumerate(segmentation):
						if PY3:
							tokenized += data[word_idx][idx]
						else:
							tokenized += data[word_idx][idx]
						if proba:
							if segmentation_probas[idx] < out_proba:
								out_proba = segmentation_probas[idx]
						if bit == 1:
							if self.enforce_allowed:
								neg_idx = -1*(len(data[word_idx])-idx-1)
								this_char = data[word_idx][idx]
								next_char = "" if idx == len(data[word_idx]) else data[word_idx][idx + 1]
								if idx not in self.allowed and neg_idx not in self.allowed:
									continue
								else:
									seg_allowed = False
									if idx in self.allowed:
										if this_char in self.allowed[idx]:
											seg_allowed = True
									if neg_idx in self.allowed:
										if next_char in self.allowed[neg_idx]:
											seg_allowed = True
									if not seg_allowed:
										continue
							tokenized += sep
			out_tokenized.append(tokenized)
			if proba:
				out_probas.append(out_proba)

		if proba:
			return out_tokenized, out_probas
		else:
			return out_tokenized


if __name__ == "__main__":
	from argparse import ArgumentParser

	parser = ArgumentParser()
	parser.add_argument("-n","--newline",action="store_true",help="insert newline instead of pipe between segments")
	parser.add_argument("-m","--model",action="store",default="cop",help="language model file path or identifier; extension .sm2/.sm3 is automatically checked for")
	parser.add_argument("-t","--train",action="store_true",help="run training")
	parser.add_argument("-l","--lexicon",action="store",default=None,help="lexicon file to use in training")
	parser.add_argument("-f","--freqs",action="store",default=None,help="frequency file to use in training")
	parser.add_argument("-c","--conf",action="store",default=None,help="configuration file to use in training")
	parser.add_argument("-i","--importances",action="store_true",help="output variable importances during test phrase of training",default=False)
	parser.add_argument("-p","--proportion",action="store",default=0.1,type=float,choices=[FloatProportion(0.0, 1.0)],help="Proportion of training data to reserve for testing")
	parser.add_argument("-e","--errors",action="store_true",help="Whether to output errors during training evaluation to errs.txt")
	parser.add_argument("-r","--retrain_all",action="store_true",help="re-run training on entire dataset (train+test) after testing")
	parser.add_argument("-a","--ablations",action="store",default=None,help="comma separated feature names to ablate in experiments")
	parser.add_argument("-o","--optimize",action="store_true",help="run hyperparameter optimization",default=False)
	parser.add_argument("-v","--version",action="store_true",help="print version number and quit")
	parser.add_argument("file",action="store",help="file to tokenize or train on")

	if "-v" in sys.argv or "--version" in sys.argv:
		print("RFTokenizer V" + __version__)
		sys.exit()

	options = parser.parse_args()

	rf_tok = RFTokenizer(options.model)

	if options.train:
		sys.stderr.write("Training...\n")
		do_dump = True
		if options.retrain_all:
			do_dump = False
		rf_tok.train(train_file=options.file, lexicon_file=options.lexicon, dump_model=do_dump, freq_file=options.freqs, output_errors=options.errors,
					 output_importances=options.importances, test_prop=options.proportion, ablations=options.ablations, conf=options.conf,
					 cross_val_test=options.optimize)
		if options.retrain_all:
			print("\no Retraining on complete data set (no test partition)...")
			rf_tok.train(train_file=options.file, lexicon_file=options.lexicon, dump_model=True, output_importances=False,
						 freq_file=options.freqs, test_prop=0.0, ablations=options.ablations, conf=options.conf)
		sys.exit()

	file_ = options.file
	data = io.open(file_, encoding="utf8").read().strip().split("\n")

	if options.newline:
		sep = "\n"
	else:
		sep = "|"

	output = rf_tok.rf_tokenize(data, sep=sep)
	if PY3:
		sys.stdout.buffer.write("\n".join(output).encode("utf8"))
	else:
		print("\n".join(output).encode("utf8"))

