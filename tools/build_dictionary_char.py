"""
Build a character dictionary

"""

import numpy
import cPickle as pkl

import sys
import fileinput

from collections import OrderedDict

def main(filenames):
    char_freqs = OrderedDict()
    for filename in filenames:
        print '==> Processing', filename
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                for c in line.strip():
                    if c not in char_freqs:
                        char_freqs[c] = 0
                    char_freqs[c] += 1
    chars = char_freqs.keys()
    freqs = char_freqs.values()
    sorted_idx = numpy.argsort(freqs)
    sorted_chars = [chars[ii] for ii in sorted_idx[::-1]]
    chardict = OrderedDict()
    chardict['eos'] = 0 
    chardict['UNK'] = 1 
    for ii, cc in enumerate(sorted_chars):
        chardict[cc] = ii+2
    with open('./char_dict.pkl', 'wb') as f:
        pkl.dump(chardict, f)
    print 'Done'

if __name__ == '__main__':
	filenames = sys.argv[1:]
	assert len(filenames) > 0, "please specify at least one filename."
	main(filenames)
