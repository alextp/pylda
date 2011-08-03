# coding: utf-8


import numpy as np
import random, os
import math
from scipy.special import gamma,gammaln
import sys
import collections
import random

import numpy as np, math, random
from scipy.special import gamma, gammaln

def discrete(probs):
    return np.argmax(np.random.multinomial(1,probs))


def gamma_pdf(x,k,theta):
    x,k,theta = map(float,(x,k,theta))
    return (x**(k-1))*(math.exp(-x/theta))/((theta**k)*gamma(k))


def exp_pdf(x,k):
    return k*math.exp(-k*x)

def dargmax(d):
    ma = -np.inf
    mv = None
    for k,v in d.iteritems():
        if ma < v:
            ma = v
            mv = k
    return mv

import re
wre = re.compile(r"(\w)+")
def get_words(text, stop=True):
    "A simple tokenizer"
    l = 0
    while l < len(text):
        s = wre.search(text,l)
        try:
            w = text[s.start():s.end()].lower()
            if stop:
                yield w
            l = s.end()
        except:
            break

def get_word_stop(text):
    "A simple tokenizer"
    l = 0
    while l < len(text):
        s = wre.search(text,l)
        try:
            w = text[s.start():s.end()]
            yield False,text[l:s.start()]
            yield True,w
            l = s.end()
        except:
            break


class LDASampler(object):
    def __init__(self, fname):
        self.all_words = []
        self.reverse_map = {}
        self.doc_map = {}
        self.documents = []
        self.topics = [] 
        self.topmap = {}
        self.topsums = {}
        self.dtopics = []
        self.alpha = 1.0
        self.beta = 1.0
        self.dsums = []
        self.Ndocuments = 0
        self.Nwords = 0
        self.load_docs(fname)
        self.Nwords = len(self.all_words)
        self.Ndocuments = len(self.documents)
        self.assignments = [[0 for w in d] for d in self.documents]
        self.mcsample = [[collections.defaultdict(int) for w in d] 
                         for d in self.documents]
        self.initialize()
    
    def load_doc(self,topics,text):
        "Creates a bag of words for a single document"
        v = []
        document = text
        for w in get_words(document):
            w = w.lower()
            if not w in self.reverse_map:
                self.reverse_map[w] = self.Nwords
                self.all_words.append(w)
                self.Nwords += 1
            v.append(self.reverse_map[w])
        if not v: return
        for topic in topics:
            if not topic in self.topmap:
                self.topmap[topic] = collections.defaultdict(int)
                self.topsums[topic] = [0]
        self.dtopics.append([self.topmap[t] for t in topics])
        self.dsums.append([self.topsums[t] for t in topics])
        self.doc_map[document] = len(self.documents)
        self.docs.append(text)
        self.documents.append(v)
        self.topics.append(topics)

    def do_inference(self, text, niter):
        document = text
        v = []
        for w in get_words(document):
            w = w.lower()
            if not w in self.reverse_map:
                self.reverse_map[w] = self.Nwords
                self.all_words.append(w)
                self.Nwords += 1
            v.append(self.reverse_map[w])
        if not v: return
        print "Document", text
        topics = self.topmap.keys()
        tcounts = [self.topmap[x] for x in topics]
        tsums = [self.topsums[x][0] for x in topics]
        assignments = [random.randint(0,len(topics)-1) for i in v]
        dcounts = np.array([self.beta for x in topics])
        samp = np.zeros_like(dcounts)
        for a in assignments: dcounts[a] += 1
        for i in xrange(niter):
            for i,w in enumerate(v):
                to = assignments[i]
                dcounts[to] -= 1
                p = np.array([dcounts[t]*(tcounts[t][w] + self.alpha)/
                              (tsums[t]+len(self.all_words)*self.alpha)
                              for t in xrange(len(topics))])
                p /= np.sum(p)
                nt = discrete(p)
                dcounts[nt] += 1
                assignments[i] = nt
            samp += dcounts
        ds = np.sum(samp)
        for topic,count in zip(topics, samp):
            print "   %3f %s" %(count/ds,topic)
        
        
    def load_docs(self, fname):
        self.Nproducts = 0
        self.prod = []
        self.docs = []
        for line in file(fname):
            labels, text = line.split(",")
            topics = [x.strip() for x in labels.split(" ")]
            self.load_doc(topics, text)

    def initialize(self):
        self.tcount = []
        for d in xrange(self.Ndocuments):
            tc = np.array([1.0 for x in self.dtopics[d]])
            tc /= np.sum(tc)
            self.tcount.append(np.array([0.0 for x in self.dtopics[d]]))
            for i,w in enumerate(self.documents[d]):
                t = discrete(tc)
                self.assignments[d][i] = t
                self.tcount[d][t] += 1
                self.dtopics[d][t][w] += 1
                self.dsums[d][t][0] += 1
        self.tcount = np.array(self.tcount)
        self.tsamp = np.zeros_like(self.tcount)

    def resample_word(self, d,i,w):
        "Resamples the topic assignments of a word"
        to = self.assignments[d][i]
        assert self.documents[d][i] == w, "%s %s %s"%(d,i,w)
        assert self.dtopics[d][to][w] > 0, "%d %d %d %s"%(d, to, w, i)
        assert self.dsums[d][to][0] > 0
        self.dtopics[d][to][w] -= 1
        self.dsums[d][to][0] -= 1
        self.tcount[d,to] -= 1
        pt = np.zeros(len(self.dtopics[d]))
        tc = self.tcount[d] + self.alpha
        for j in xrange(len(pt)):
            pt[j] = (tc[j]*(self.dtopics[d][j][w]+self.beta)/
                     (self.dsums[d][j][0]+self.beta*self.Nwords))
        pt /= np.sum(pt)
        nt = discrete(pt)
        self.assignments[d][i] = nt
        self.dtopics[d][nt][w] += 1
        self.dsums[d][nt][0] += 1
        self.tcount[d,nt] += 1


    def iterate(self):
        for document in xrange(self.Ndocuments):
            for i,word in enumerate(self.documents[document]):
                self.resample_word(document,i,word)
        self.tsamp += self.tcount
        sys.stdout.flush()

    def fit(self,its):
        "The sampler itself."
        iteration = 0
        print "iterating.."
        for i in xrange(its):
            iteration += 1
            self.iterate()
        for name,counts in self.topmap.items():
            print name
            print_keyw_topic(self, counts, 40)
            print
            
    

def print_topic(model, t, n):
    s = np.argsort(-t)
    for w in s[:n]:
        print "     ",model.all_words[w], t[w]


def print_keyw_topic(model, t, n):
    tt = np.zeros(len(model.all_words))
    for k,v in t.items():
        tt[k] = v
    print_topic(model, tt, n)

def top_keyw_topic(model, t, n):
    tt = np.zeros(len(model.all_words))
    for k,v in t.items():
        tt[k] = v
    s = np.argsort(-tt)
    return [model.all_words[i] for i in s[:n]]


if __name__ == '__main__':
    NTOPICS = 1
    import sys
    print sys.argv[1:]
    s = LDASampler(sys.argv[1])
    s.fit(150)
    for line in file(sys.argv[2]):
        s.do_inference(line, 100)

