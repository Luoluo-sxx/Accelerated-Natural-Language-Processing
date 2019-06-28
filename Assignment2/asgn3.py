from __future__ import division
from math import log,sqrt
import operator
from nltk.stem import *
from nltk.stem.porter import *
import matplotlib.pyplot as plt
from load_map import *
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
STEMMER = PorterStemmer()

# helper function to get the count of a word (string)
def w_count(word):
  return o_counts[word2wid[word]]

def tw_stemmer(word):
  '''Stems the word using Porter stemmer, unless it is a 
  username (starts with @).  If so, returns the word unchanged.

  :type word: str
  :param word: the word to be stemmed
  :rtype: str
  :return: the stemmed word

  '''
  if word[0] == '@': #don't stem these
    return word
  else:
    return STEMMER.stem(word)

def PMI(c_xy, c_x, c_y, N, a=1):
  '''Compute the pointwise mutual information using cooccurrence counts.

  :type c_xy: int 
  :type c_x: int 
  :type c_y: int 
  :type N: int
  :param c_xy: coocurrence count of x and y
  :param c_x: occurrence count of x
  :param c_y: occurrence count of y
  :param N: total observation count
  :rtype: float
  :return: the pmi value

  '''
  return log(c_xy * (N) / (c_x * (c_y ** a)), 2)#0 # you need to fix this

def cos_sim(v0,v1):
  '''Compute the cosine similarity between two sparse vectors.

  :type v0: dict
  :type v1: dict
  :param v0: first sparse vector
  :param v1: second sparse vector
  :rtype: float
  :return: cosine between v0 and v1
  '''
  # We recommend that you store the sparse vectors as dictionaries
  # with keys giving the indices of the non-zero entries, and values
  # giving the values at those dimensions.

  s = 0.
  l0 = 0.
  for i in v0:
      l0 += v0[i]**2
      if i in v1:
          s += v0[i] * v1[i]
  l1 = 0.
  for i in v1:
      l1 += v1[i]**2
  return s / (sqrt(l0) * sqrt(l1))




def create_ppmi_vectors(wids, o_counts, co_counts, tot_count, smoothing=False, a = 0.75):
    '''Creates context vectors for the words in wids, using PPMI.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''

    vectors = {}
    if smoothing:
        s = tot_count**a
    for wid0 in wids:
        vectors[wid0] = {}
        for wid1 in co_counts[wid0].keys():
            c_xy = co_counts[wid0][wid1]
            if not smoothing:
                pmi = PMI(c_xy, o_counts[wid0],o_counts[wid1], tot_count, a=1)
            else:
                pmi = PMI(c_xy, o_counts[wid0],o_counts[wid1], s, a=a)
            if pmi > 0:
                vectors[wid0][wid1] = pmi
    return vectors


def read_counts(filename, wids):
  '''Reads the counts from file. It returns counts for all words, but to
  save memory it only returns cooccurrence counts for the words
  whose ids are listed in wids.

  :type filename: string
  :type wids: list
  :param filename: where to read info from
  :param wids: a list of word ids
  :returns: occurence counts, cooccurence counts, and tot number of observations
  '''
  o_counts = {} # Occurence counts
  co_counts = {} # Cooccurence counts
  fp = open(filename)
  N = float(next(fp))
  for line in fp:
    line = line.strip().split("\t")
    wid0 = int(line[0])
    o_counts[wid0] = int(line[1])
    if(wid0 in wids):
        co_counts[wid0] = dict([int(y) for y in x.split(" ")] for x in line[2:])
  return (o_counts, co_counts, N)

def print_sorted_pairs(similarities, o_counts, first=0, last=100):
  '''Sorts the pairs of words by their similarity scores and prints
  out the sorted list from index first to last, along with the
  counts of each word in each pair.

  :type similarities: dict 
  :type o_counts: dict
  :type first: int
  :type last: int
  :param similarities: the word id pairs (keys) with similarity scores (values)
  :param o_counts: the counts of each word id
  :param first: index to start printing from
  :param last: index to stop printing
  :return: none
  '''
  if first < 0: last = len(similarities)
  for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = True)[first:last]:
    word_pair = (wid2word[pair[0]], wid2word[pair[1]])
    print("{:.2f}\t{:30}\t{}\t{}".format(similarities[pair],str(word_pair),
                                         o_counts[pair[0]],o_counts[pair[1]]))

#for eculidean distance 
def print_sorted_pairs_2(similarities, o_counts, first=0, last=100):
  '''Sorts the pairs of words by their similarity scores and prints
  out the sorted list from index first to last, along with the
  counts of each word in each pair.

  :type similarities: dict 
  :type o_counts: dict
  :type first: int
  :type last: int
  :param similarities: the word id pairs (keys) with similarity scores (values)
  :param o_counts: the counts of each word id
  :param first: index to start printing from
  :param last: index to stop printing
  :return: none
  '''
  if first < 0: last = len(similarities)
  for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = False)[first:last]:
    word_pair = (wid2word[pair[0]], wid2word[pair[1]])
    print("{:.2f}\t{:30}\t{}\t{}".format(similarities[pair],str(word_pair),
                                         o_counts[pair[0]],o_counts[pair[1]]))




def freq_v_sim(sims,title):
  xs = []
  ys = []
  for pair in sims.items():
    ys.append(pair[1])
    c0 = o_counts[pair[0][0]]
    c1 = o_counts[pair[0][1]]
    xs.append(min(c0,c1))
  plt.clf() # clear previous plots (if any)
  plt.xscale('log') #set x axis to log scale. Must do *before* creating plot
  plt.plot(xs, ys, 'k.') # create the scatter plot
  plt.xlabel('Min Freq')
  plt.ylabel('Similarity')
  plt.title(title+"Freq vs Similarity Spearman correlation = {:.2f}".format(stats.spearmanr(xs,ys)[0]))
  print("Freq vs Similarity Spearman correlation = {:.2f}".format(stats.spearmanr(xs,ys)[0]))
  print("Freq vs Similarity Pearson correlation = {:.2f}".format(stats.pearsonr(xs,ys)[0]))
#  plt.show() #display the set of plots

def make_pairs(items):
  '''Takes a list of items and creates a list of the unique pairs
  with each pair sorted, so that if (a, b) is a pair, (b, a) is not
  also included. Self-pairs (a, a) are also not included.

  :type items: list
  :param items: the list to pair up
  :return: list of pairs

  '''
  return [(x, y) for x in items for y in items if x < y]


#to make the data points which closed to the origin sparser 
def euclidean_log(v0, v1):
    s = 0
    for i in v0:
        if i in v1:
            s += (log(v0[i],10) - log(v1[i],10)) ** 2
        else:
            s += log(v0[i],10) ** 2
            
    for i in v1:
        if i in v0:
            continue
        else:
            s += log(v1[i],10) ** 2
        
    return sqrt(s)

def euclidean(v0, v1):
    s = 0
    for i in v0:
        if i in v1:
            s += (v0[i] -v1[i] )** 2
        else:
            s += v0[i] ** 2
            
    for i in v1:
        if i in v0:
            continue
        else:
            s += v1[i] ** 2
        
    return sqrt(s)


#transform dictionary into matrix
def dict_matrix(d):
    m = np.zeros(shape = (len(d), 29172245))
    rcount = 0
    wordw = []
    for i in d:
        wordw.append(i)
        for j in d[i]:
            m[rcount][j] = d[i][j]
        rcount += 1
    return m, wordw

#tranform matrix into dictionary
def matrix_dict(m, wordw):
    vector = {}
    rcount = 0
    for i in wordw:
        d_v = {}
        for j in range(m.shape[1]):
            if m[rcount][j] == 0:
                continue
            else:
                d_v[j] = m[rcount][j]
        vector[i] = d_v
        rcount += 1
    return vector

#PCA
def pca_reduce(m, K):
    pca = PCA(n_components=K)
    return pca.fit_transform(m)

def pearson_correlation(s1, s2):
    s1_v = []
    s2_v = []
    for i in s1:
        s1_v.append(s1[i])
        s2_v.append(s2[i])
    s1_v = np.array(s1_v)
    s2_v = np.array(s2_v)
    s1_v_mean = s1_v.mean()
    s2_v_mean = s2_v.mean()
    s1_v_var = s1_v.var()
    s2_v_var = s2_v.var()
    #print(np.dot((s1_v - s1_v_mean).T, s2_v - s2_v_mean))
    
    return np.dot((s1_v - s1_v_mean).T, s2_v - s2_v_mean) / (np.sqrt(s1_v_var * s2_v_var) * s1_v.shape[0])

def spearman_correlation(s1, s2):
    s1_v = []
    s2_v = []
    for i in s1:
        s1_v.append(s1[i])
        s2_v.append(s2[i])
    s1_v.sort()
    s2_v.sort()
    s1_v = np.array(s1_v)
    s2_v = np.array(s2_v)
    s1_v_mean = s1_v.mean()
    s2_v_mean = s2_v.mean()
    s1_v_var = s1_v.var()
    s2_v_var = s2_v.var()
    #print(np.dot((s1_v - s1_v_mean).T, s2_v - s2_v_mean))
    
    return np.dot((s1_v - s1_v_mean).T, s2_v - s2_v_mean) / (np.sqrt(s1_v_var * s2_v_var) * s1_v.shape[0])

def Jaccard(v0, v1):
    n = 0.
    d = 0.
    for i in v0:
        if i in v1:
            n += min(v0[i], v1[i])
            d += max(v0[i], v1[i])
        else:
            d += v0[i]
    
    for i in v1:
        if i not in v0:
            d += v1[i]
            
    return n / d

def Dice(v0, v1):
    n = 0.
    for i in v0:
        if i in v1:
            n += 1
    return 2 * n / (len(v0) + len(v1))

test_result=[ 'apricot','avocado','banana','blackberri','blackcurrant','blueberri',
               'cherri','coconut','fig','grape','grapefruit','kiwi'
               ,'lemon','lime','lychee','mandarin','mango','melon','nectarine',
               'orange','papaya','peach','pear','pineapple'
               ,'plum','pomegranate','quince','raspberri','strawberri','watermelon','orange','olive','coconut','eggplant',
               'wine','carrot','cucumber','pepper','mushroom','potato','lobster','sugar',
               'sandwich','rice','soup','dumpling','bread','sausage','pizza','cake','ham',
               'cheese','brandy','sherry','soda','juice','beer','vodka','milk','tomato','ginger','pumpkin',
               'yum','tasty','hungry','starving','nice','good','pretty','beautiful','bad',
               'terrible','smart','clever','happy','sad','lucky','wonderful','excited',
               'upset','brave','honey','sweet','honest','dishonest','evil','pure',
               'disappointed','mad','desperate','crazy',
               'love','hate','feel','follow','play','watch','today','hope','wait','haha','yeah','friend','life',
                'orange','coffee','purple','white','red','green','blue','copper','silver',
                'gold','rainbow','yellow','black','grey','pink','coral','brown','violet',
                'indigo',
                "@justinbieber","@selenagomez","@missravensymone","@jlo","@ninadobrev","@apple","@microsoft","@sony",
                '#egypt','#tigerblood','#threewordstoliveby',
                '#idontunderstandwhy','#japan','#superbowl','#jan25','#someday',
                '#biggieday','#auckland', '#theman','#winner',
                'cat','dog','bird','mouse','tiger','giraffe','lion','deer','monkey','elephant',
               'kangaroo','shrimp','fly','butterfly','parrot','swan','eagle','fish',
               'mosquito','donkey','hedgehog','horse','pig','chicken','snake','rabbit',
               'fox','pandas','squirrel'
                
               ]



test_words = ["cat", "dog", "mouse", "computer","@justinbieber"]
stemmed_words = [tw_stemmer(w) for w in test_result]
all_wids = set([word2wid[x] for x in stemmed_words]) #stemming might create duplicates; remove them

# you could choose to just select some pairs and add them by hand instead
# but here we automatically create all pairs 
wid_pairs = make_pairs(all_wids)


#read in the count information
(o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/counts", all_wids)
'''
stemmed_words = [tw_stemmer(w) for w in number]
for i in stemmed_words:
    try :
        co_counts[word2wid[i]]
    except:
        print(i)
        stemmed_words.remove(i)
'''
#Do a simple error check using value computed by hand
if(PMI(2,4,3,12) != 1): # these numbers are from our y,z example
    print("Warning: PMI is incorrectly defined")
else:
    print("PMI check passed")
#make the word vectors
#=========
#vectors = tf_idf(all_wids, o_counts, co_counts, N)

vectors = create_ppmi_vectors(all_wids, o_counts, co_counts, N, False)
vectors_s = create_ppmi_vectors(all_wids, o_counts, co_counts, N, True)
# compute cosine similarites for all pairs we consider

#==================
#pca visualization
'''
m, n = dict_matrix(vectors)
m = pca_reduce(m, 2)

plt.scatter(m[:,0], m[:,1], marker='x')
for l,i in zip(vectors, m):
    plt.annotate(wid2word[l], xy = i, xytext = i+0.005) #plt.plot(m, 'rx', label = l)
plt.show()



#==================
'''

c_sims1 = {(wid0,wid1): euclidean_log(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
c_sims2 = {(wid0,wid1): cos_sim(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
c_sims3 = {(wid0,wid1): Jaccard(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
c_sims4 = {(wid0,wid1): Dice(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
c_sims5 = {(wid0,wid1): euclidean_log(vectors_s[wid0],vectors_s[wid1]) for (wid0,wid1) in wid_pairs}
c_sims6 = {(wid0,wid1): Jaccard(vectors_s[wid0],vectors_s[wid1]) for (wid0,wid1) in wid_pairs}
c_sims7 = {(wid0,wid1): Dice(vectors_s[wid0],vectors_s[wid1]) for (wid0,wid1) in wid_pairs}
c_sims9 = {(wid0,wid1): cos_sim(vectors_s[wid0],vectors_s[wid1]) for (wid0,wid1) in wid_pairs}

'''
print("Sort by euclidean similarity")
print_sorted_pairs_2(c_sims1, o_counts)
'''
print("Sort by cosine")
print_sorted_pairs(c_sims2, o_counts)
freq_v_sim(c_sims2,'cosine')
plt.savefig("cosine.pdf")
'''
print("Sort by cosine similarity")
print_sorted_pairs(c_sims2, o_counts)
print("Sort by Jaccard similarity")
print_sorted_pairs(c_sims3, o_counts)
print("Sort by Dice similarity")
print_sorted_pairs(c_sims4, o_counts)
print("Sort by euclidean similarity with smoothing")
print_sorted_pairs_2(c_sims5, o_counts)
print("Sort by Jaccard similarity with smoothing")
print_sorted_pairs(c_sims6, o_counts)
print("Sort by Dice similarity with smoothing")
print_sorted_pairs(c_sims7, o_counts)
print("Sort by Dice similarity with co_counts")
print_sorted_pairs(c_sims8, o_counts)
print('Pearson correlation:', pearson_correlation(c_sims1, c_sims5))
print('Spearman correlation:', spearman_correlation(c_sims1, c_sims5))
#print('Pearson correlation:', pearson_correlation(c_sims2, c_sims4))
freq_v_sim(c_sims1)
'''
'''
low_frequency =['apricot','avocado','fig', 'grapefruit','kiwi','lychee','mandarin','melon','nectarin','papaya','plum','pomegran',
                'quinc','dumpling','@missravensymone','@jlo','@apple','@microsoft','@sony']
high_frequency = ['love','hate','feel','follow','play','watch','today','hope','wait','haha','yeah','friend','life','yo']

sport = ["pingpong","basketball","football","soccer","badminton","boxing"]
country = ['japan','china','austria','chinese','japanese']
all_res =["basketball",'love','like','happy','good','nice','hate','sad','mad','terribl','bad']
pos_word =['love','like','happy','good','nice']
neg_word =['hate','sad','mad','terribl','bad']
fruit_words = ['apricot','avocado','banana','blackberri','blackcurrant','blueberri',
               'cherri','coconut','fig','grape','grapefruit','kiwi'
               ,'lemon','lime','lychee','mandarin','mango','melon','nectarine',
               'orange','papaya','peach','pear','pineapple'
               ,'plum','pomegranate','quince','raspberri','strawberri','watermelon','orange','olive','coconut','eggplant']
'''