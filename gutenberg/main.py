from __future__ import print_function
import itertools
import nltk
import sys
import os
import logging


vocabulary_size = 3000
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
START_TOKEN = "SENTENCE_START"
END_TOKEN = "SENTENCE_END"


def write_sentences(sentences, fname):
    with open(fname, 'w') as fhdl:
        for x in sentences:
            line = (x + '\n').encode('utf-8')
            fhdl.write(line)
    return


def load_sentencesX(fname):
    logging.info("begin to loadd sentences from %s" % (fname))
    sentences = []
    paragraph = ""
    pcount = 0

    fhdl = open(fname, 'rb')
    for line in fhdl:
        line = line.decode('utf-8').lower()
        line = line.strip("\r\n")
        if len(line) < 1:
            sentences = itertools.chain(sentences, nltk.sent_tokenize(paragraph))
            paragraph = ""
            pcount += 1
            continue

        paragraph += line

    if len(paragraph) > 2:
        sentences = itertools.chain(sentences, nltk.sent_tokenize(paragraph))
        pcount += 1
    fhdl.close()

    sentences = list(sentences)
    # sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    logging.info("%d paragraphs, %d sentences" % (pcount, len(sentences)))
    return sentences


def build_dict(sentences, fname_dict, fname_dat):
    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    logging.info("Found %d unique words tokens", len(word_freq.items()))

    # 1. get vocab
    vocab = word_freq.most_common(vocabulary_size - 3)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(START_TOKEN)
    index_to_word.append(END_TOKEN)
    index_to_word.append(UNKNOWN_TOKEN)
    lastword = vocab[-1]
    logging.info("Vocab size %d, last word is '%s', freq is %s", len(index_to_word), lastword[0], lastword[1])

    # 2. write the dict
    with open(fname_dict, 'wb') as fdict:
        for i in range(len(index_to_word)):
            fdict.write("%d %s\n" % (i, index_to_word[i].encode('utf-8')))

    return tokenized_sentences


def load_dict(fname_dict):
    word_to_index = dict()
    with open(fname_dict, 'rb') as fdict:
        for line in fdict:
            items = line.strip().split(" ")
            word = items[1].decode('utf-8')
            word_to_index[word] = int(items[0])

    logging.info("%d words in dict" % (len(word_to_index)))
    return word_to_index


def replace_unknown(worddict, sentences):
    result = []
    for x in sentences:
        x = [START_TOKEN] + x + [END_TOKEN]
        nx = [w if w in worddict else UNKNOWN_TOKEN for w in x]
        result.append(nx)
    return result


def gen_train_dat(sentences, fname_dict, fname_raw_dat, fname_train):
    word_to_index = load_dict(fname_dict)
    fhdl_raw = open(fname_raw_dat, 'wb')
    fhdl_train = open(fname_train, 'wb')

    sentences = replace_unknown(word_to_index, sentences)

    for x in sentences:
        words = x 

        # write the dat for training
        nums = [word_to_index[w] for w in words]
        nline = " ".join(str(x) for x in nums)
        fhdl_train.write("%s\n" % (nline))

        # write out the raw dat
        line = ""
        for w in words:
            line += (w.encode('utf-8') +  " ")
        fhdl_raw.write("%s\n" % (line))

    fhdl_train.close()
    fhdl_raw.close()
    return


class ResultFnames:
    def __init__(self, basedir, corpfname, prefix):
        self.corpfname = corpfname
        self.sentence_fname = os.path.join(basedir, prefix + ".sentence.dat")
        self.dict_fname = os.path.join(basedir, prefix + ".dict.dat")
        self.raw_fname = os.path.join(basedir, prefix + ".raw.train.dat")
        self.train_fname = os.path.join(basedir, prefix + ".train.dat")
        self.tmp_fname = os.path.join(basedir, prefix + ".tmp.dat")
        return

    def __str__(self):
        corpf = "\tcorp_fname: %s\n" % (self.corpfname)
        dictf = "\tdict_fname: %s\n" % (self.dict_fname) 
        rawf = "\traw_train_fname: %s\n" % (self.raw_fname)
        train = "\ttrain_fname: %s" % (self.train_fname)
        msg = "\n" + corpf + dictf + rawf + train
        return msg


def main(args):
    setup_log()
    basedir = "./data/"
    corpfname = "../data/alice.txt"
    #corpfname = "../data/whitman-leaves.txt"
    prefix = 'alice'
    if len(sys.argv) > 1:
        corpfname = sys.argv[1]
        bname = os.path.basename(corpfname)
        prefix, _ = os.path.splitext(bname)

    logging.info("input fname=%s, prefix=%s", corpfname, prefix)
    fnames = ResultFnames(basedir, corpfname, prefix)
    #fnames = ResultFnames(basedir, corpfname, 'whitman')
    logging.info("fnames: %s", fnames)

    sentences = load_sentencesX(fnames.corpfname)
    write_sentences(sentences, fnames.sentence_fname)
    word_sentences = build_dict(sentences, fnames.dict_fname, fnames.tmp_fname)
    gen_train_dat(word_sentences, fnames.dict_fname, fnames.raw_fname, fnames.train_fname)
    return

def setup_log():
    logfile = "./log/train.%s.log" % (os.getpid())
    print("logfile=%s"%(logfile))
    logging.basicConfig(#filename=logfile
            format='[%(asctime)s.%(msecs)d] %(levelname)-s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%d-%m-%Y:%H:%M:%S',
            level=logging.INFO)
    return


if __name__ == "__main__":
    sys.exit(main(sys.argv))
