"""
Simple utility to convert conllu files to tab-separated files with segments
"""

import io, os, sys, re
from glob import glob

def get_segs(conllu):
    super_length = 0
    limit = 10  # Maximum bound group length in units, discard sentences with longer groups
    sents = []
    words = []
    labels = []
    word = []
    max_len = 0
    lines = conllu.split("\n")
    for l, line in enumerate(lines):
        if "\t" in line:
            fields = line.split("\t")
            if "-" in fields[0]:
                start, end = fields[0].split("-")
                super_length = int(end) - int(start) + 1
            else:
                if super_length > 0:
                    word.append(fields[1])
                    super_length -= 1
                    if super_length == 0:
                        words.append("".join(word))
                        labels.append("|".join(word))
                        if len(word) > max_len:
                            max_len = len(word)
                        word = []
                else:
                    if "SpaceAfter=No" in line and ("ADP\t" in line or "DET\t" in line):
                        done = False
                        word.append(fields[1])
                        counter = 1
                        while not done:
                            if "SpaceAfter" in lines[l+counter] and not ("\t,\t" in lines[l+counter+1] or "\t.\t" in lines[l+counter+1]):
                                super_length += 1
                                counter += 1
                            else:
                                super_length += 1
                                done = True
                                if super_length > 10:
                                    print(l)
                                    quit()
                    else:
                        words.append(fields[1])
                        labels.append(fields[1])
        elif len(line) == 0 and len(words) > 0:
            if max_len > limit or " " in "".join(words):  # Reject sentence
                max_len = 0
            else:
                sents.append("\n".join([w + "\t" + l for w, l, in zip(words, labels)]))
            words = []
            labels = []
    return "\n".join(sents)

files = glob("*.conllu")

for file_ in files:
    seg_data = get_segs(io.open(file_).read())
    with io.open(os.path.basename(file_) + ".tab",'w',encoding="utf8",newline="\n") as f:
        f.write(seg_data)
