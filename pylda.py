# coding: utf-8

"""
Implements gibbs sampling for the Latent Dirichlet Allocation using
the algorithm presented by Griffiths and Steyvers "Finding Scientific
Topics".

The main difference is that instead of marginalizing many samples we
choose one single sample after the burn-in interval to compute the
relevant statistics."""

import numpy as np
import random
import math
from scipy.special import gamma,gammaln
from scipy import weave
import sys

import re
wre = re.compile(r"(\w)+")
def get_words(text):
    "A simple tokenizer"
    l = 0
    while l < len(text):
        s = wre.search(text,l)
        try:
            yield text[s.start():s.end()]
            l = s.end()
        except:
            break


all_words = []
reverse_map = {}
def get_all_words(docs):
    "Run this before running make_bag to preload the vocabulary"
    for d in docs:
        for w in get_words(d):
            w = w.lower()
            if not w in reverse_map:
                reverse_map[w] = len(all_words)
                all_words.append(w)


def make_bag(document):
    "Creates a bag of words for a single document"
    v = []
    for w in get_words(document):
        w = w.lower()
        if not w in reverse_map:
            reverse_map[w] = len(all_words)
            all_words.append(w)
        v.append(reverse_map[w])
    return v

def categorical2(probs):
    return np.argmax(np.random.multinomial(1,probs))

def cond_dist(d,i,w,assignments, documents,Nwt,Ntd,Nwtcs,Ntdcs, pa, pb,
              Ntopics,Nwords,Ndocuments,alpha,beta):
    "Samples the conditional distribution for the assignment of a word to a topic."
    to = assignments[d][i]
    Nwtcs[to] -= 1
    Ntdcs[d] -= 1
    Nwt[w,to] -= 1
    Ntd[d,to] -= 1
    aa = (Nwt[w]+beta)
    bb = (Nwtcs[to]+pb)
    cc = (Ntd[d]+alpha)
    dd = (Ntdcs[d]+pa)
    pt = (aa/bb)*(cc/dd)
    pt /= np.sum(pt)
    nt = categorical2(pt)
    assignments[d][i] = nt
    Nwtcs[nt] += 1
    Ntdcs[d] += 1
    Nwt[w,nt] += 1
    Ntd[d,nt] += 1
    return pt[nt]

def phi(assignments, bags, Ntopics, Nwords, Ndocuments, alpha, beta):
    p = beta*np.ones((Ntopics,Nwords)) 
    for d in xrange(Ndocuments):
        for i,w in enumerate(bags[d]):
            t = assignments[d][i]
            p[t,w] += 1
    return p

def theta(assignments, bags, Ntopics, Nwords, Ndocuments, alpha, beta):
    th = alpha*np.ones((Ndocuments,Ntopics))
    for d in xrange(Ndocuments):
        for i,w in enumerate(bags[d]):
            t = assignments[d][i]
            th[d,t] += 1
    return th

def likelihood(assignments, Nwtcs, Ntdcs, bags, Ntopics, Nwords, Ndocuments, 
               alpha, beta):
    "Computes the likelihood of the parameters"
    f1 = Ndocuments*(gammaln(Ntopics*alpha)-Ntopics*gammaln(alpha))
    vt = np.zeros(Ntopics)
    f2 = 0.
    for d in xrange(Ndocuments):
        vt.fill(0)
        for i,w in enumerate(bags[d]):
            vt[assignments[d][i]] += 1
        vt += alpha
        f2t1 = np.sum(gammaln(vt))
        f2t2 = gammaln(Ntdcs[d]+Ntopics*alpha)
        f2 += f2t1-f2t2
    return f1 + f2
    
def mean(x):
    return sum(x)/len(x)

def run(Ntopics,alpha,beta,bags,interval,nsamples):
    "The sampler itself."
    Nwords = len(all_words)
    Ndocuments = len(bags)
    assignments = [[0 for w in d] for d in bags]
    Nwt = np.zeros((Nwords,Ntopics))
    Ntd = np.zeros((Ndocuments,Ntopics))
    Nwtcs = np.zeros(Ntopics)
    Ntdcs = np.zeros(Ndocuments)
    old_lik = -np.inf
    sampling = False
    samples = []
    for d in xrange(Ndocuments):
        for i,w in enumerate(bags[d]):
            t = random.randint(0,Ntopics-1)
            assignments[d][i] = t
            Nwt[w,t] += 1
            Ntd[d,t] += 1
            Nwtcs[t] += 1
            Ntdcs[d] += 1
    pa = alpha*Nwords
    pb = beta*Ntopics
    iteration = 0
    while iteration < 20:#len(samples) <  nsamples:
        iteration += 1
        for document in xrange(Ndocuments):
            for i,word in enumerate(bags[document]):
                pp = cond_dist(document,i,word,assignments,bags,Nwt,Ntd,Nwtcs,Ntdcs,
                               pa,pb,Ntopics,Nwords,Ndocuments,alpha, beta)
        lik = likelihood(assignments,Nwtcs, Ntdcs,bags,Ntopics,Nwords,Ndocuments,
                         alpha,beta)
        print lik
        if lik- old_lik < 0 and not sampling: 
            sampling = True
            print "Now sampling"
        old_lik = lik
        if sampling:
            if iteration % interval != 0: continue
            p,t = (phi(assignments, bags, Ntopics, Nwords, Ndocuments, alpha, beta), 
            theta(assignments, bags, Ntopics, Nwords, Ndocuments, alpha, beta))
            samples.append((p,t))
    return mean([a[0] for a in samples]), mean([a[1] for a in samples])

def print_topic(phi, t):
    print "topico", t,":"
    s = np.argsort(-phi[t])
    for w in s[:20]:
        print "     ",all_words[w]


def print_topics(phi):
    for t in xrange(len(phi)):
        print_topic(phi,t)
        print

def make_reverse_map():
    for i,w in enumerate(all_words):
        reverse_map[w] = i

def parse_lda_data(prefix):
    data_f = file(prefix+".data")
    vocab = [a.strip() for a in file(prefix+".vocab")]
    global all_words
    all_words = vocab
    make_reverse_map()
    Nwords = len(vocab)
    data = [a.strip().split() for a in data_f]
    Ndocuments = len(data)
    documents = [[] for i in xrange(Ndocuments)]
    for doc in xrange(Ndocuments):
        for word in data[doc][1:]:
            w,c = word.split(":")
            [documents[doc].append(w) for i in xrange(c)]
    return documents
    

def test(word, documents):
    import svm,random
    docs = [d.copy() for d in documents if d[reverse_map[word]]]
    nondocs = [d.copy() for d in documents if not d[reverse_map[word]]]
    nondocs = random.sample(nondocs,min(5*len(docs),len(nondocs)))
    print float(len(nondocs))/(len(docs)+len(nondocs))
    cats = [1 for i in docs] + [0 for i in nondocs]
    obs = docs + nondocs
    for i in xrange(len(obs)):
        obs[i][reverse_map[word]] = 0.
    zobs = zip(obs,cats)
    random.shuffle(zobs)
    obs,cats = zip(*zobs)
    params = svm.svm_parameter(C=1, kernel_type=svm.LINEAR)
    problem = svm.svm_problem(cats,obs)
    target = svm.cross_validation(problem,params,20)
    return sum(target[i] == cats[i] for i in cats)/float(len(cats))

        
f = ("/home/top/textos/Douglas Adams/Douglas Adams -"
     " So Long, and Thanks For All the Fish.txt")

if __name__=='__main__':
    get_all_words([file(f).read()])
    documents = [make_bag(x) for x in file(f).read().split("\r\n\r\n")]
    phi,theta = run(10, 1.,1., documents,4, 5)
    print "returned"
    print_topics(phi)
    
    
