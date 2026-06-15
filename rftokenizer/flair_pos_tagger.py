"""
flair_pos_tagger.py

This module trains flair sequence labelers to predict POS and deprel for OTHER modules.
"""


from argparse import ArgumentParser
import flair
from flair.data import Corpus, Sentence, Dictionary
from flair.datasets import ColumnCorpus
from flair.embeddings import OneHotEmbeddings, WordEmbeddings, CharacterEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
import os, sys, io
from glob import glob
from random import seed, shuffle
PY3 = sys.version_info[0] > 2

if PY3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

seed(42)

flair_version = int(flair.__version__.split(".")[1])

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
model_dir = script_dir + ".." + os.sep + "models" + os.sep
CONLLU_ROOT = "conllu" + os.sep  # Path to UD .conllu corpus repo directory
TARGET_FEATS = {}  # If using this tagger for specific features, specify them here


class FlairTagger:

    def __init__(self, train=False, morph=False, seg=False, lang="heb"):
        self.lang = lang  # Prefix for the language name in the model, e.g. heb for Hebrew
        if not train:
            if morph:
                self.model = SequenceTagger.load(model_dir + lang + ".morph")
            elif seg:
                model_dir = script_dir + "models" + os.sep
                if not os.path.exists(model_dir + lang + ".seg"):
                    sys.stderr.write("! Model file " + model_dir + lang + ".seg not found\n")
                    sys.stderr.write("! Attempting to download it... (this could take a while)\n")
                    url = "https://gucorpling.org/amir/download/heb_models_v4/" + lang + ".seg"
                    urlretrieve(url, model_dir + lang + ".seg")
                    sys.stderr.write("! Done!\n")
                self.model = SequenceTagger.load(model_dir + lang + ".seg")
            else:
                self.model = SequenceTagger.load(model_dir + lang + ".flair")

    def make_seg_data(self):
        if self.lang == "heb":
            # Pre-determined prefixes and suffixes for Hebrew, can be changed for other language training
            prefixes = {"ב","כ","מ","ל","ה",}
            suffixes = {"ו","ה","י","ך","ם","ן","הם","הן","כם","כן","יו"}
            conjunctions = {"ש","כש"}
            preprefixes = {"ו"}
        elif self.lang == "cop":
            prefixes = {"ⲡ","ⲡⲓ","ⲫ","ⲡⲁⲓ","ⲫⲁⲓ","ⲡⲉⲓ","ⲡⲉ","ⲧ","ϯ","ⲑ","ⲧⲁⲓ","ⲑⲁⲓ","ⲧⲉⲓ","ⲧⲉ","ⲛ","ⲙ","ⲛⲓ","ⲛⲁⲓ","ⲛⲉⲓ","ⲛⲉ","ⲟⲩ","ⲩ","ϩⲉⲛ","ϩⲁⲛ","ⲡⲉϥ","ⲡⲁ",""}
            suffixes = {"ⲓ","ⲕ","ϥ","ⲥ","ⲧⲛ","ⲧⲉⲛ","ⲛⲧⲉⲧⲛ","ⲧⲉⲧⲉⲛ","ⲧⲉⲧⲛ"}
            conjunctions = {"ש","כש"}
            preprefixes = {"ו"}

        def segs2tag(segs, tags=None):
            if tags is None:
                tag = "X"
                if len(segs) == 2:
                    if segs[0] in preprefixes:
                        tag = "W"
                    elif segs[0] in conjunctions:
                        tag = "S"
                    elif segs[0] in prefixes:
                        tag = "B"
                    if segs[1] in suffixes:
                        tag += "Y"
                elif len(segs) == 3:
                    if segs[0] in preprefixes:
                        tag = "W"
                    elif segs[0] in conjunctions:
                        tag = "S"
                    elif segs[0] in prefixes:
                        tag = "B"
                    if segs[1] in conjunctions:
                        tag += "S"
                    elif segs[1] in prefixes:
                        tag += "B"
                    if segs[2] in suffixes:
                        tag += "Y"
                elif len(segs) > 3:
                    if segs[0] in preprefixes:
                        tag = "W"
                    elif segs[0] in conjunctions:
                        tag = "S"
                    if segs[1] in conjunctions:
                        tag += "S"
                    elif segs[1] in prefixes:
                        tag += "B"
                    if segs[2] in prefixes:
                        tag += "B"
                    if segs[-1] in suffixes:
                        tag += "Y"
                if tag == "BS":
                    tag = "BB"  # מ+ש, כ+ש
                elif tag == "WSY":  # ושעיקרה
                    tag = "WBY"
                elif "XS" in tag:
                    tag = "X"
                return tag.replace("SS","S")
            else:
                tag = []
                if len(tags) == 1:
                    return "_"
                if segs[0] in ["ϫⲉ","ⲛϭⲓ","ⲛϫⲉ"]:
                    tag.append("J")
                if any([t in ["CCIRC","CFOC","CREL","CPRET"] for t in tags]):
                    tag.append("C")
                #if any([t.startswith("NEG") for t in tags]):
                #    tag.append("N")
                if any([t.startswith("A") and t not in ["ACAUS","ADV","ART"] for t in tags]):
                    tag.append("A")
                if any(["PPERS" in t for t in tags]):
                    tag.append("S")
                if any([t in ["ART","PDEM","PPOS"] for t in tags]):
                    tag.append("D")
                if any(["PREP" in t for t in tags]):
                    tag.append("P")
                if any(["PPERO" in t for t in tags]):
                    tag.append("O")
                if len(tag) > 4:
                    return "X"  # Long sequence
                if "".join(tag) in ["JO","JP","ASDP","JADP","CPO","ADPO","JASD","JCAS","CP","CSPO","CDP","CADO","JSO","SP","AO","JDP","DPO","ADO","JCD","CSP","JCSO","APO","CSD","JC","JASP","SD","CSDP","CAO","ASDO","SPO","CASP","JAP","JPO","JADO","DO","CAP","JAO","SDP","JSD","JCSD","JSPO","JCSP","CDPO","JAPO","JCA","JCO","JDO","SDO"]:
                    return "Y"  # Rare tag
                return "".join(tag) if len(tag) > 0 else "_"

        def conllu2segs(conllu, target="affixes", limit=4, tags=False):
            # param: limit - Maximum bound group length in units, discard sentences with longer groups
            super_length = 0
            sents = []
            words = []
            pos = []
            labels = []
            word = []
            max_len = 0
            lines = conllu.split("\n")
            for line in lines:
                if "\t" in line:
                    fields = line.split("\t")
                    if "-" in fields[0]:
                        start, end = fields[0].split("-")
                        super_length = int(end) - int(start) + 1
                    else:
                        if super_length > 0:
                            word.append(fields[1])
                            pos.append(fields[4])
                            super_length -= 1
                            if super_length == 0:
                                words.append("".join(word))
                                if target=="count":
                                    labels.append(str(len(word)))
                                else:
                                    if tags:
                                        labels.append(segs2tag(word, pos))
                                    else:
                                        labels.append(segs2tag(word))
                                if len(word) > max_len:
                                    max_len = len(word)
                                word = []
                                pos = []
                        else:
                            words.append(fields[1])
                            labels.append("O")
                elif len(line) == 0 and len(words) > 0:
                    if max_len > limit or " " in "".join(words):  # Reject sentence
                        max_len = 0
                    else:
                        sents.append("\n".join([w + "\t" + l for w, l, in zip(words,labels)]))
                    words = []
                    labels = []
            return "\n\n".join(sents)

        root = CONLLU_ROOT
        if self.lang == "heb":
            root = CONLLU_ROOT + "seg"
        files = glob(root + os.sep + "*.conllu")
        data = ""
        for file_ in files:
            if self.lang == "heb":
                # For Hebrew, a special affix scheme is preconfigured - adapt for other languages or use generic count
                data += conllu2segs(io.open(file_, encoding="utf8").read(),target="affixes", limit=4) + "\n\n"
            elif self.lang == "cop":
                # For Coptic, labels are based on POS tag sequences
                data += conllu2segs(io.open(file_, encoding="utf8").read(), target="affixes", limit=8, tags=True) + "\n\n"
            else:
                data += conllu2segs(io.open(file_,encoding="utf8").read(), target="count") + "\n\n"
        sents = data.strip().split("\n\n")
        sents = list(set(sents))
        shuffle(sents)
        with io.open("tagger" + os.sep + self.lang + "_train_seg.txt", 'w', encoding="utf8",newline="\n") as f:
            f.write("\n\n".join(sents[:int(-len(sents)/10)]))
        with io.open("tagger" + os.sep + self.lang + "_dev_seg.txt", 'w', encoding="utf8",newline="\n") as f:
            f.write("\n\n".join(sents[int(-len(sents)/10):]))
        with io.open("tagger" + os.sep + self.lang + "_test_seg.txt", 'w', encoding="utf8",newline="\n") as f:
            f.write("\n\n".join(sents[int(-len(sents)/10):]))

    def make_pos_data(self, tags=False):
        def filter_morph(feats):
            if feats == "_":
                return "O"
            else:
                annos = []
                for f in feats.split("|"):
                    k, v = f.split("=")
                    if k in TARGET_FEATS:
                        annos.append(k+"="+v)
                if len(annos) > 0:
                    return "|".join(annos)
                else:
                    return "O"

        files = glob(CONLLU_ROOT + "*.conllu")
        train = test = dev = ""
        super_tok_len = 0
        super_tok_start = False
        suff = "_morph" if tags else ""
        for file_ in files:
            output = []
            lines = io.open(file_,encoding="utf8").readlines()
            for line in lines:
                if "\t" in line:
                    fields = line.split("\t")
                    if "." in fields[0]:
                        continue
                    if "-" in fields[0]:
                        super_tok_start = True
                        start,end = fields[0].split("-")
                        super_tok_len = int(end)-int(start) + 1
                        continue
                    if super_tok_start:
                        super_tok_position = "B"
                        super_tok_start = False
                        super_tok_len -= 1
                    elif super_tok_len > 0:
                        super_tok_position = "I"
                        super_tok_len -= 1
                        if super_tok_len == 0:
                            super_tok_position = "E"
                    else:
                        super_tok_position = "O"
                    if tags:
                        morph = filter_morph(fields[5])
                        output.append(fields[1] + "\t" + super_tok_position + "\t" + fields[4] + "\t" + morph)
                    else:
                        output.append(fields[1] + "\t" + super_tok_position + "\t" + fields[4])
                elif len(line.strip()) == 0:
                    if output[-1] != "":
                        output.append("")
            if "dev" in file_:
                dev += "\n".join(output)
            elif "test" in file_:
                test += "\n".join(output)
            else:
                train += "\n".join(output)
        with io.open("tagger" + os.sep + self.lang + "_train"+suff+".txt", 'w', encoding="utf8",newline="\n") as f:
            f.write(train)
        with io.open("tagger" + os.sep + self.lang + "_dev"+suff+".txt", 'w', encoding="utf8",newline="\n") as f:
            f.write(dev)
        with io.open("tagger" + os.sep + self.lang + "_test"+suff+".txt", 'w', encoding="utf8",newline="\n") as f:
            f.write(test)

    def train(self, cuda_safe=True, positional=True, tags=False, seg=False):
        if cuda_safe:
            # Prevent CUDA Launch Failure random error, but slower:
            import torch
            torch.backends.cudnn.enabled = False
            # Or:
            # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        # 1. get the corpus
        # this is the folder in which train, test and dev files reside
        data_folder = "tagger" + os.sep

        # init a corpus using column format, data folder and the names of the train, dev and test files

        # define columns
        columns = {0: "text", 1: "super", 2: "pos"}
        suff = ""
        if positional:
            columns[1] = "super"
            columns[2] = "pos"
        if tags:
            columns[3] = "morph"
            suff = "_morph"
        if seg:
            columns[1] = "seg"
            del columns[2]
            self.make_seg_data()
            suff = "_seg"
        else:
            self.make_pos_data(tags=tags)

        corpus: Corpus = ColumnCorpus(
            data_folder, columns,
            train_file=self.lang + "_train"+suff+".txt",
            test_file=self.lang + "_test"+suff+".txt",
            dev_file=self.lang + "_dev"+suff+".txt",
        )

        # 2. what tag do we want to predict?
        tag_type = 'pos' if not tags else "morph"
        if seg:
            tag_type = "seg"

        # 3. make the tag dictionary from the corpus
        if flair_version > 8:
            tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)
        else:
            tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
        print(tag_dictionary)

        # 4. initialize embeddings
        # Set language specific transformer embeddings here
        if self.lang == "heb":
            embeddings: TransformerWordEmbeddings = TransformerWordEmbeddings('onlplab/alephbert-base',)  # AlephBERT for Hebrew
        else:
            embeddings: TransformerWordEmbeddings = TransformerWordEmbeddings('lgessler/microbert-coptic-mx',)
        if positional:
            positions: OneHotEmbeddings = OneHotEmbeddings(corpus=corpus, field="super", embedding_length=5)
            if tags:
                tag_emb: OneHotEmbeddings = OneHotEmbeddings(corpus=corpus, field="pos", embedding_length=17)
                stacked: StackedEmbeddings = StackedEmbeddings([embeddings,positions,tag_emb])
            else:
                stacked: StackedEmbeddings = StackedEmbeddings([embeddings, positions])
        elif not seg:
            if tags:
                tag_emb: OneHotEmbeddings = OneHotEmbeddings(corpus=corpus, field="pos", embedding_length=17)
                stacked: StackedEmbeddings = StackedEmbeddings([embeddings,tag_emb])
            else:
                stacked = embeddings
        else:
            if self.lang == "cop":
                w2v: WordEmbeddings = WordEmbeddings(script_dir + os.sep + "coptic_50d_norm_and_group_sb.vec.gensim")
                char: CharacterEmbeddings = CharacterEmbeddings(Dictionary.load("chars-large"))
                stacked: StackedEmbeddings = StackedEmbeddings([embeddings, w2v, char])
            else:
                stacked = embeddings

        # 5. initialize sequence tagger
        hidden_size = 256 if self.lang == "heb" else 128

        tagger: SequenceTagger = SequenceTagger(hidden_size=hidden_size,
                                                embeddings=stacked,
                                                tag_dictionary=tag_dictionary,
                                                tag_type=tag_type,
                                                use_crf=False,
                                                use_rnn=False)

        # 6. initialize trainer
        from flair.trainers import ModelTrainer

        trainer: ModelTrainer = ModelTrainer(tagger, corpus)

        # 7. start training
        trainer.train(script_dir + "pos-dependencies" + os.sep + 'flair_tagger',
                      learning_rate=0.1,
                      mini_batch_size=24,
                      max_epochs=150)

    def predict(self, in_path=None, in_format="flair", out_format="conllu", as_text=False, tags=False, seg=False):
        model = self.model
        tagcol = 4

        if as_text:
            data = in_path
            #data = (data + "\n").replace("<s>\n", "").replace("</s>\n", "\n").strip()
        else:
            data = io.open(in_path,encoding="utf8").read()
        sents = []
        words = []
        positions = []
        true_tags = []
        true_pos = []
        super_tok_start = False
        super_tok_len = 0
        data = data.strip() + "\n"  # Ensure final new line for last sentence
        for line in data.split("\n"):
            if len(line.strip())==0:
                if len(words) > 0:
                    if flair_version > 8:
                        tokenizer = False
                    else:
                        tokenizer = lambda x:x.split(" ")
                    sents.append(Sentence(" ".join(words),use_tokenizer=tokenizer))
                    for i, word in enumerate(sents[-1]):
                        if not seg:
                            word.add_label("super",positions[i])
                        if tags:
                            word.add_label("pos",true_pos[i])
                    words = []
                    positions = []
                    true_pos = []
            else:
                if in_format == "flair":
                    words.append(line.split("\t")[0])
                    if not seg:
                        positions.append(line.split("\t")[1])
                    if tags:
                        true_pos.append(line.split("\t")[2])
                        true_tags.append(line.split("\t")[3]) if "\t" in line else true_tags.append("")
                    else:
                        true_tags.append(line.split("\t")[2]) if "\t" in line else true_tags.append("")
                else:
                    if "\t" in line:
                        fields = line.split("\t")
                        if "." in fields[0]:
                            continue
                        if "-" in fields[0]:
                            super_tok_start = True
                            start, end = fields[0].split("-")
                            super_tok_len = int(end) - int(start) + 1
                            continue
                        if super_tok_start:
                            super_tok_position = "B"
                            super_tok_start = False
                            super_tok_len -= 1
                        elif super_tok_len > 0:
                            super_tok_position = "I"
                            super_tok_len -= 1
                            if super_tok_len == 0:
                                super_tok_position = "E"
                        else:
                            super_tok_position = "O"
                        words.append(line.split("\t")[1])
                        positions.append(super_tok_position)
                        true_tags.append(line.split("\t")[tagcol])
                        true_pos.append(line.split("\t")[4])

        # predict tags and print
        if flair_version > 8:
            model.predict(sents, force_token_predictions=True, return_probabilities_for_all_classes=True)
        else:
            model.predict(sents)  # , all_tag_prob=True)

        preds = []
        scores = []
        words = []
        for i, sent in enumerate(sents):
            for tok in sent.tokens:
                if tags:
                    pred = tok.labels[2].value
                    score = str(tok.labels[2].score)
                elif seg:
                    if flair_version > 8:
                        pred = tok.labels[0].value if len(tok.labels)>0 else "O"
                        score = tok.labels[0].score if len(tok.labels) > 0 else "1.0"
                    else:
                        label = tok.labels[0]
                        pred = label.value
                        score = str(label.score)
                else:
                    pred = tok.labels[1].value
                    score = str(tok.labels[1].score)
                preds.append(pred)
                scores.append(score)
                words.append(tok.text)

        toknum = 0
        output = []
        #out_format="diff"
        for i, sent in enumerate(sents):
            tid=1
            if i>0 and out_format=="conllu":
                output.append("")
            for tok in sent.tokens:
                pred = preds[toknum]
                score = str(scores[toknum])
                if len(score)>5:
                    score = score[:5]
                if out_format == "conllu":
                    pred = pred if not pred == "O" else "_"
                    fields = [str(tid),tok.text,"_",pred,pred,"_","_","_","_","_"]
                    output.append("\t".join(fields))
                    tid+=1
                elif out_format == "xg":
                    output.append("\t".join([pred, tok.text, score]))
                else:
                    true_tag = true_tags[toknum]
                    corr = "T" if true_tag == pred else "F"
                    output.append("\t".join([pred, true_tag, corr, score, tok.text, true_pos[toknum]]))
                toknum += 1

        if as_text:
            return "\n".join(output)
        else:
            ext = "xpos.conllu" if out_format == "conllu" else "txt"
            partition = "test" if "test" in in_path else "dev"
            with io.open(script_dir + "pos-dependencies" +os.sep + "flair-"+partition+"-pred." + ext,'w',encoding="utf8",newline="\n") as f:
                f.write("\n".join(output))


if __name__ == "__main__":
    # To train a segmentation category predictor for RFTokenizer set the conllu root above, choose
    # the transformer in TransformerWordEmbeddings above and use:
    # python flair_pos_tagger.py --seg -i conllu
    p = ArgumentParser()
    p.add_argument("-m","--mode",choices=["train","predict"],default="predict")
    p.add_argument("-l","--lang",default="heb",help="Language prefix")
    p.add_argument("-f","--file",default=None,help="Blank for training, blank predict for eval, or file to run predict on")
    p.add_argument("-p","--positional_embeddings",action="store_true",help="Whether to use positional embeddings within supertokens (MWTs)")
    p.add_argument("-t","--tag_embeddings",action="store_true",help="Whether to use POS tag embeddings for morphology prediction")
    p.add_argument("-s","--seg",action="store_true",help="Whether to train segmentation instead of tagging")
    p.add_argument("-i","--input_format",choices=["flair","conllu"],default="flair",help="flair two column training format or conllu")
    p.add_argument("-o","--output_format",choices=["flair","conllu","xg"],default="conllu",help="flair two column training format or conllu")

    opts = p.parse_args()

    if opts.mode == "train":
        tagger = FlairTagger(train=True,lang=opts.lang)
        tagger.train(positional=opts.positional_embeddings, tags=opts.tag_embeddings, seg=opts.seg)
    else:
        tagger = FlairTagger(train=False,lang=opts.lang)
        tagger.predict(in_format=opts.input_format, out_format=opts.output_format,
                in_path=opts.file, seg=opts.seg)
