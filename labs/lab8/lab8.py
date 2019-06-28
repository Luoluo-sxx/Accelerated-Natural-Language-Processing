'''
Authors: Luke Shrimpton, Sharon Goldwater, Ida Szubert
Date: 2014-11-01, 2017-11-05
Copyright: This work is licensed under a Creative Commons
Attribution-NonCommercial 4.0 International License
(http://creativecommons.org/licenses/by-nc/4.0/): You may re-use,
redistribute, or modify this work for non-commercial purposes provided
you retain attribution to any previous author(s).
'''
from __future__ import division;
from math import log;
from pylab import mean;
from load_map import *
import numpy as np

def PMI(c_xy, c_x, c_y, N):
    # Computes PMI(x, y) where
    # c_xy is the number of times x co-occurs with y
    # c_x is the number of times x occurs.
    # c_y is the number of times y occurs.
    # N is the number of observations.
    p_xy = float(c_xy) /N
    p_x = float(c_x) /N
    p_y = float(c_y) /N
    pmi = log(p_xy/(p_x*p_y),2)
    return pmi; # replace this

#Do a simple error check using value computed by hand
if(PMI(2,4,3,12) != 1): # these numbers are from our y,z example
    print("Warning: PMI is incorrectly defined")
else:
    print("PMI check passed")

# List of positive words:
pos_words = ["love"];
# List of negative words:
neg_words = ["hate"];
# List of target words:
targets = ["@justinbieber","husband","wife","son","daughter","kid","child","facebook", "school","food", "toothbrush","bf","gf"];

# Collect all words of interest and store their term ids:
all_words = set(pos_words+neg_words+targets);
all_wids = set([word2wid[x] for x in all_words]);

# Define the data structures used to store the counts:
o_counts = {}; # Occurrence counts
co_counts = {}; # Co-occurrence counts

# Load the data:
fp = open("/afs/inf.ed.ac.uk/group/teaching/anlp/lab8/counts", "r");
lines = fp.readlines();
N = float(lines[0]); # First line contains the number of observations.
for line in lines[1:]:
    line = line.strip().split("\t");
    wid0 = int(line[0]);
    if(wid0 in all_wids): # Only get/store counts for words we are interested in
        o_counts[wid0] = int(line[1]); # Store occurence counts
        co_counts[wid0] = dict([[int(y) for y in x.split(" ")] for x in line[2:]]); # Store co-occurence counts


# This code currently does nothing, students will fill in
for target in targets:
    targetid = word2wid[target]
    posPMIs = []
    negPMIs = []
    # compute PMI between target and each positive word, and
    # add it to the list of positive PMI values
    for pos in pos_words:
        posid = word2wid[pos]
        pmi =PMI(co_counts[posid][targetid],o_counts[targetid],o_counts[targetid],N)
        posPMIs.append(pmi)
    # same for negative words
    for neg in neg_words:
        negid = word2wid[neg]
        pmi =PMI(co_counts[negid][targetid],o_counts[targetid],o_counts[targetid],N)
        negPMIs.append(pmi)
#uncomment the following line when posPMIs and negPMIs are no longer empty.
    print(target, ": ", mean(posPMIs), "(pos), ", mean(negPMIs), "(neg)")

   
