#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys, os, io, random, re
import numpy as np
import pandas as pd
if __name__ == "__main__":
	from preprocess import LetterConfig
else:
	from .preprocess import LetterConfig
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from collections import defaultdict
try:
	from ConfigParser import RawConfigParser as configparser
except ImportError:
	from configparser import RawConfigParser as configparser

PY3 = sys.version_info[0] == 3
script_dir = os.path.dirname(os.path.realpath(__file__))

def lambda_underscore():  # Module level named 'lambda' function to make defaultdict picklable
	return "_"

class FloatProportion(object):
	def __init__(self, start, end):
		self.start = start
		self.end = end
	def __eq__(self, other):
		return self.start <= other <= self.end

def read_lex(short_pos, lex_file):
	"""
	Read a tab delimited lexicon file. The first two columns must be word-form and POS tag.

	:param short_pos: Dictionary possibly mapping POS tags in lexicon to alternative (usually collapsed) POS tags.
	:param lex_file: Name of file to read.
	:return: defaultdict returning the concatenated POS tags of all possible analyses of each form, or "_" if unknown.
	"""
	lex_lines = io.open(lex_file, encoding="utf8").read().replace("\r", "").split("\n")
	lex = defaultdict(set)

	for line in lex_lines:
		if "\t" in line:
			word, pos = line.split("\t")[0:2]
			if pos in short_pos:
				lex[word].add(short_pos[pos])
			else:
				lex[word].add(pos)

	pos_lookup = defaultdict(lambda_underscore)
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

	def __init__(self, model="cop"):
		self.loaded = False
		self.model = model
		self.lang = os.path.basename(model).replace(".sm2","").replace(".sm3","")
		self.conf = {}
		self.regex_tok = None
		self.short_pos = {}
		self.pos_lookup = defaultdict(lambda: "_")
		self.conf["base_letters"] = set()
		self.conf["vowels"] = set()
		config = configparser()
		if not os.path.isfile(script_dir + os.sep + "models.conf"):
			sys.stderr.write("FATAL: could not find configuration file models.conf in " + script_dir + "\n")
			sys.exit()
		try:
			config.read_file(io.open(script_dir + os.sep + "models.conf",encoding="utf8"))
		except AttributeError:
			config.readfp(io.open(script_dir + os.sep + "models.conf",encoding="utf8"))
		if not config.has_section(self.lang):
			sys.stderr.write("FATAL: could not find section [" + self.lang + "] in models.conf in " + script_dir + "\n")
			sys.stderr.write("       File should contain section [" + self.lang + "] and at least setting for included 'letters')\n")
			sys.exit()
		for key, val in config.items(self.lang):
			if key in ["base_letters","vowels"]:
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
						sys.stderr.write("WARN: regex entry without tab in models.conf\n")
		self.letters = defaultdict(lambda : self.conf["base_letters"])

	def load(self, model_path=None):
		"""
		Load a picked model.

		:param model_path: Path to the model pickle file. If not specified, looks for model language name +.sm2 (Python 2) or .sm3 (Python 3), e.g. heb.sm3
		:return: void
		"""
		if model_path is None:
			# Default model path for a language is the language name, extension ".sm2" for Python 2 or ".sm3" for Python 3
			model_path = self.lang + ".sm" + str(sys.version_info[0])
		if PY3:
			(self.tokenizer, self.num_labels, self.cat_labels, self.encoder, self.preparation_pipeline, self.pos_lookup) = joblib.load(model_path)
		else:
			(self.tokenizer, self.num_labels, self.cat_labels, self.encoder, self.preparation_pipeline, self.pos_lookup) = joblib.load(model_path)

	def train(self, train_file, lexicon_file=None, test_prop=0.1, output_importances=False, dump=False, cross_val_test=False, output_errors=False):
		import timing

		pos_lookup = read_lex(self.short_pos,lexicon_file)
		self.pos_lookup = pos_lookup
		letter_config = LetterConfig(self.letters, self.conf["vowels"], self.pos_lookup)

		np.random.seed(42)

		if lexicon_file is None:
			print("i WARN: No lexicon file provided, learning purely from examples")

		seg_table = io.open(train_file,encoding="utf8").read()
		seg_table = seg_table.replace("\r","").strip().split("\n")

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
		gold_segmentations = []
		all_encoded_groups = []

		encoding_cache = {}
		non_ident_segs = 0

		shuffle_mapping = list(range(len(seg_table)))
		zipped = list(zip(seg_table, shuffle_mapping))

		random.Random(24).shuffle(zipped)

		seg_table, shuffle_mapping = zip(*zipped)

		headers = bg2array("_________",prev_group="_",next_group="_",print_headers=True,is_test=1,grp_id=1,config=letter_config,lang=self.lang)

		word_idx = -1
		bug_rows = []

		step = int(1/test_prop) if test_prop > 0 else 0
		test_indices = list(range(len(seg_table)))[0::step] if step > 0 else []

		for row_idx, row in enumerate(seg_table):
			is_test = 1 if row_idx in test_indices else 0

			prev_group, next_group, bound_group, segmentation = row.split("\t")
			if len(bound_group) != len(segmentation.replace("|","")):  # Ignore segmentations that also normalize
				non_ident_segs += 1
				bug_rows.append(row_idx)
				continue
			gold_segmentations.append(segmentation)
			word_idx += 1
			words.append(bound_group)
			group_type = "_".join([x for x in [prev_group, next_group, bound_group] if x != ""])
			if group_type in encoding_cache:
				encoded_group = encoding_cache[group_type]
			else:
				encoded_group = bg2array(bound_group,prev_group=prev_group,next_group=next_group,is_test=is_test,grp_id=word_idx,config=letter_config,lang=self.lang,train=True)
				encoding_cache[group_type] = encoded_group
			all_encoded_groups += encoded_group
			data_y += segs2array(segmentation)

		sys.stderr.write("o Finished encoding " + str(len(data_y)) + " chars (" + str(len(seg_table)) + " groups, " + str(len(encoding_cache)) + " group types)\n")

		if non_ident_segs > 0:
			with open("bug_rows.txt",'w') as f:
				f.write("\n".join([str(r) for r in sorted([shuffle_mapping[x] for x in bug_rows])]) + "\n")

			sys.stderr.write("i WARN: found " + str(non_ident_segs) + " rows in training data where left column characters not identical to right column characters\n")
			sys.stderr.write("        Row numbers dumped to: buggy_rows.txt\n")
			sys.stderr.write("        " + str(non_ident_segs) + " rows were ignored in training\n\n")

		data_y = np.array(data_y)

		cat_labels = ['group_in_lex','current_letter', 'prev_prev_letter', 'prev_letter', 'next_letter', 'next_next_letter',
					 'mns4_coarse', 'mns3_coarse', 'mns2_coarse',
					 'mns1_coarse', 'pls1_coarse', 'pls2_coarse',
					 'pls3_coarse', 'pls4_coarse', "so_far_pos", "remaining_pos","prev_grp_pos","next_grp_pos",
					  "remaining_pos_mns1","remaining_pos_mns2"]

		num_labels = ['idx','len_bound_group',"current_vowel","prev_prev_vowel","prev_vowel","next_vowel","next_next_vowel"]

		# Add context labels
		cat_labels += ["prev_grp_first","prev_grp_last"]
		num_labels += ["prev_grp_len"]
		cat_labels += ["next_grp_first","next_grp_last"]
		num_labels += ["next_grp_len"]

		data_x = pd.DataFrame(all_encoded_groups, columns=headers)

		encoder = MultiColumnLabelEncoder(pd.Index(cat_labels))
		data_x_enc = encoder.fit_transform(data_x)

		if test_prop > 0:
			sys.stderr.write("o Generating train/test split with test proportion "+str(test_prop)+"\n")

		data_x_enc["boundary"] = data_y
		strat_train_set = data_x_enc.iloc[data_x_enc.index[data_x_enc["is_test"] == 0]]
		strat_test_set = data_x_enc.iloc[data_x_enc.index[data_x_enc["is_test"] == 1]]

		cat_pipeline = Pipeline([
			('selector', DataFrameSelector(cat_labels)),
		])

		num_pipeline = Pipeline([
			('selector', DataFrameSelector(num_labels)),
			# ('std_scaler', StandardScaler()),
		])

		preparation_pipeline = FeatureUnion(transformer_list=[
			("cat_pipeline", cat_pipeline),
			("num_pipeline", num_pipeline),
		])

		sys.stderr.write("o Transforming data to numerical array\n")
		train_x = preparation_pipeline.fit_transform(strat_train_set)

		train_y = strat_train_set["boundary"]
		train_y_bin = np.where(strat_train_set['boundary'] == 0, 0, 1)

		if test_prop > 0:
			test_x = preparation_pipeline.transform(strat_test_set)
			test_y_bin = np.where(strat_test_set['boundary'] == 0, 0, 1)
			bound_grp_idx = np.array(strat_test_set['grp_id'])

			from sklearn.dummy import DummyClassifier
			d = DummyClassifier(strategy="most_frequent")
			d.fit(train_x,train_y_bin)
			pred = d.predict(test_x)
			print("o Majority baseline:")
			print("\t" + str(accuracy_score(test_y_bin, pred)))

		forest_clf = ExtraTreesClassifier(n_estimators=200, max_features=None, n_jobs=-1, random_state=42)

		if cross_val_test:
			# Modify code to tune hyperparameters/use different estimators

			from sklearn.model_selection import GridSearchCV
			sys.stderr.write("o Running CV...\n")

			params = {"n_estimators":[300,400,500],"max_features":["auto",None]}#,"class_weight":["balanced",None]}
			grid = GridSearchCV(RandomForestClassifier(n_jobs=-1,random_state=42,warm_start=True),param_grid=params,refit=False)
			grid.fit(train_x,train_y_bin)
			print("\nGrid search results:\n" + 30 * "=")
			for key in grid.cv_results_:
				print(key + ": " + str(grid.cv_results_[key]))

			print("\nBest parameters:\n" + 30 * "=")
			print(grid.best_params_)
			sys.exit()

		sys.stderr.write("o Learning...\n")
		forest_clf.fit(train_x, train_y_bin)

		if test_prop > 0:
			pred = forest_clf.predict(test_x)
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

			if output_importances:
				feature_names = cat_labels + num_labels

				zipped = zip(feature_names, forest_clf.feature_importances_)
				sorted_zip = sorted(zipped, key=lambda x: x[1], reverse=True)
				print("o Feature importances:\n")
				for name, importance in sorted_zip:
					print(name, "=", importance)
		else:
			print("o Test proportion is 0%, skipping evaluation")

		if dump:
			joblib.dump((forest_clf, num_labels, cat_labels, encoder, preparation_pipeline, pos_lookup), self.lang + ".sm" + str(sys.version_info[0]), compress=3)
			print("o Dumped trained model to " + self.lang + ".sm" + str(sys.version_info[0]))


	def rf_tokenize(self, data, sep="|", indices=None):
		"""
		Main tokenizer routine

		:param data: ordered list of word forms (prev/next word context is taken from list, so meaningful order is assumed)
		:param sep: separator to use for found segments, default: |
		:param indices: options; list of integer indices to process. If supplied, positions not in the list are skipped
		:return: list of word form strings tokenized using the separator
		"""

		if not self.loaded:
			if os.path.isfile(self.model):
				self.load(self.model)
			else:
				if os.path.isfile(self.lang + ".sm" + str(sys.version_info[0])):
					self.load(self.lang + ".sm" + str(sys.version_info[0]))
				elif os.path.isfile(script_dir + os.sep + self.lang + ".sm" + str(sys.version_info[0])):
					self.load(script_dir + os.sep + self.lang + ".sm" + str(sys.version_info[0]))
				else:
					sys.stderr.write("FATAL: Could not find segmentation model at " + script_dir + os.sep + self.model + ".sm" + str(sys.version_info[0]))
					sys.exit()
			self.loaded = True

		tokenizer, num_labels, cat_labels, encoder, preparation_pipeline = self.tokenizer, self.num_labels, self.cat_labels, self.encoder, self.preparation_pipeline

		do_not_tok_indices = set()

		if indices is not None:
			if len(indices) == 0:
				return []

		encoded_groups = []

		headers = bg2array("_________",prev_group="_",next_group="_",print_headers=True,config=LetterConfig(),lang=self.lang)
		word_lengths = []
		cursor = 0

		letters = {}
		for header in headers:
			if header in encoder.encoder_dict:
				letters[header] = encoder.encoder_dict[header].classes_

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

			encoded_group = bg2array(word,prev_group=prev_group,next_group=next_group,config=letter_config,lang=self.lang)
			encoded_groups += encoded_group
			word_lengths.append(cursor + len(word))
			cursor += len(word)
		word_lengths.pop()

		if indices is not None:
			data = [data[i] for i in indices]

		data_x = pd.DataFrame(encoded_groups)
		data_x.columns = headers

		encoder.transform(data_x)
		prepped = preparation_pipeline.transform(data_x)

		p = tokenizer.predict(prepped)
		p_words = np.split(p, word_lengths)
		out_tokenized = []
		for word_idx, segmentation in enumerate(p_words):
			tokenized = ""
			if data[word_idx] == "":
				tokenized = ""
			else:
				for idx, bit in enumerate(segmentation):
					if word_idx in do_not_tok_indices:
						word = data[word_idx][idx]
						for f, r in self.regex_tok:
							word = f.sub(r,word)
						tokenized += word
						continue
					if PY3:
						tokenized += data[word_idx][idx]
					else:
						tokenized += data[word_idx][idx]
					if bit == 1:
						tokenized += sep
			out_tokenized.append(tokenized)

		return out_tokenized


if __name__ == "__main__":
	from preprocess import DataFrameSelector, MultiColumnLabelEncoder, bg2array, segs2array
	from argparse import ArgumentParser

	parser = ArgumentParser()
	parser.add_argument("-n","--newline",action="store_true",help="insert newline instead of pipe between segments")
	parser.add_argument("-m","--model",action="store",default="cop",help="language model file path or identifier; extension .sm2/.sm3 is automatically checked for")
	parser.add_argument("-t","--train",action="store_true",help="run training")
	parser.add_argument("-i","--importances",action="store_true",help="output variable importances during test phrase of training",default=False)
	parser.add_argument("-p","--proportion",action="store",default=0.1,type=float,choices=[FloatProportion(0.0, 1.0)],help="Proportion of training data to reserve for testing")
	parser.add_argument("-r","--retrain_all",action="store_true",help="re-run training on entire dataset (train+test) after testing")
	parser.add_argument("-l","--lexicon",action="store",default=None,help="lexicon file to use in training")
	parser.add_argument("file",action="store",help="file to tokenize or train on")

	options = parser.parse_args()

	rf_tok = RFTokenizer(options.model)

	if options.train:
		sys.stderr.write("Training...\n")
		do_dump = True
		if options.retrain_all:
			do_dump = False
		rf_tok.train(train_file=options.file, lexicon_file=options.lexicon, dump=do_dump, output_importances=options.importances, test_prop=options.proportion)
		if options.retrain_all:
			print("\no Retraining on complete data set (no test partition)...")
			rf_tok.train(train_file=options.file, lexicon_file=options.lexicon, dump=True, output_importances=False, test_prop=0.0)
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

else:
	from modules.preprocess import DataFrameSelector, MultiColumnLabelEncoder, bg2array, segs2array
