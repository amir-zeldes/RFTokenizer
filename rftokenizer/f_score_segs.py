#!/usr/bin/python
# -*- coding: utf-8 -*-

import io, sys, argparse

"""
Script to evaluate segmentation f-score and perfect super-token segmentation proportion from two files:
 * goldfile: single column file, one super-token perline, pipes denote segmentation positions	
	* predfile: same format
	
Stripped files are assumed to have the same amount of characters except pipes. A super-token may also consist
of a single pipe (i.e. input contained a literal pipe, which must be a complete super-token).
"""


def main(goldfile, predfile, preds_as_string=False):
	lines = io.open(goldfile, encoding="utf8").read().strip().replace("\r", "").split("\n")
	counter = 0
	gold = []
	gold_groups = []
	perfect = 0
	total = 0

	if "\t" in lines[0]:  # Convenience step allowing 2-column file as gold
		sys.stderr.write("o Found tab in gold file, using second column as gold\n")
		lines = [line.split("\t")[1] for line in lines if "\t" in line]
	for r, line in enumerate(lines):
		total += 1
		gold_groups.append(line.strip())
		for i, c in enumerate(list(line.strip())):
			counter += 1
			if i == len(line.strip()) - 1:  # Last character is trivial, ignore
				continue
			if c == "|":
				if len(line.strip()) > 1 and i == 0:
					print("Complex token begins with boundary marker at gold row " + str(r))
					sys.exit()
				counter -=1
				gold[-1] = 1
			else:
				gold.append(0)

	gold_chars = counter

	if preds_as_string:
		lines = predfile.split("\n")
	else:
		lines = io.open(predfile, encoding="utf8").read().strip().replace("\r", "").split("\n")
	counter = 0
	pred = []

	for r, line in enumerate(lines):
		if line.strip() == gold_groups[r]:
			perfect += 1
		for i, c in enumerate(list(line.strip())):
			counter += 1
			if i == len(line.strip()) - 1:  # Last character is trivial, ignore
				continue
			if c == "|":
				if len(line.strip()) > 1 and i == 0:
					print("Complex token begins with boundary marker at pred row " + str(r))
					sys.exit()
				counter -=1
				pred[-1] = 1
			else:
				pred.append(0)

	pred_chars = counter

	if pred_chars != gold_chars:
		sys.stderr.write("ERROR: found " + str(gold_chars) + " gold chars but " + str(pred_chars) + " pred chars\n")
		sys.exit()

	true_positive = 0
	false_positive = 0
	false_negative = 0
	for i in range(len(gold)):
		if gold[i] == 0:
			if pred[i] == 0:
				pass
			else:
				false_positive += 1
		else:
			if pred[i] == 0:
				false_negative += 1
			else:
				true_positive += 1

	try:
		precision = true_positive / (float(true_positive) + false_positive)
	except Exception as e:
		precision = 0

	try:
		recall = true_positive / (float(true_positive) + false_negative)
	except Exception as e:
		recall = 0

	try:
		f_score = 2 * (precision * recall) / (precision + recall)
	except:
		f_score = 0

	try:
		perf_score = perfect/float(total)
	except:
		perf_score = 0

	print("Total chars: " + str(pred_chars))
	print("Perfect word forms: " + str(perf_score))
	print("Precision: " + str(precision))
	print("Recall: " + str(recall))
	print("F-Score: " + str(f_score))


if __name__ == "__main__":
	p = argparse.ArgumentParser()

	p.add_argument("goldfile")
	p.add_argument("predfile")

	opts = p.parse_args()

	main(opts.goldfile,opts.predfile)
