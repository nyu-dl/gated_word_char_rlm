"""
Build a word dictionary 

"""

import numpy
import cPickle as pkl

import sys
import fileinput

from collections import OrderedDict

def main(filenames):
    word_freqs = OrderedDict()
    for filename in filenames:
        print '==> Processing', filename 
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                words_in = line.lower().strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1
    words = word_freqs.keys()
    freqs = word_freqs.values()
    sorted_idx = numpy.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]
    worddict = OrderedDict()
    worddict['eos'] = 0
    worddict['UNK'] = 1  # use <unk> for PTB
    for ii, ww in enumerate(sorted_words):
        if ww != 'UNK':  # use <unk> for PTB
            worddict[ww] = ii+2
    with open('./word_dict.pkl', 'wb') as f:
        pkl.dump(worddict, f)
    print 'Done'

if __name__ == '__main__':
	filenames = sys.argv[1:]
	assert len(filenames) > 0, "please specify at least one filename."
	main(filenames)
