# coding: utf-8


import numpy as np
import random, os
import math
from scipy.special import gamma,gammaln
import sys
import collections

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


def slice_sample(likelihood, x0):
    old_lik = likelihood(x0)
    old_alpha = x0
    lnt = old_lik - np.random.exponential(1)
    w = old_alpha/32.
    L = max(0, old_alpha - w*random.random())
    R = L + w
    K = 4
    while K > 0 and (lnt < likelihood(L) or lnt < likelihood(R)):
        V = random.random()
        if V < 0.5:
            if L-(R-L) < 0:
                print "L would be", L-(R-L), "R is", R
            L = max(0, L-(R-L))
        else:
            R = R+(R-L)
        K = K-1
    rej = True
    while rej:
        U = random.random()
        x1 = L+U*(R-L)
        rr = likelihood(x1)
        if lnt < rr:
            break
        else:
            if x1 < old_alpha:
                L = x1
            else:
                R = x1
    return x1


def cl(tprior, gmean, gvar, tcount):
    "Can deal with a vector of alphas"
    p0 = sum(np.log(gamma_pdf(tp, gmean, gvar)) for tp in tprior)
    p0 += tcount.shape[0]*(gammaln(sum(tprior))-sum(gammaln(tp) for tp in tprior))
    for d in xrange(tcount.shape[0]):
        for i,t in enumerate(tcount[d]):
            p0 += gammaln(t + tprior[i])
        p0 -= gammaln(sum(tcount[d]) + sum(tprior))
    return p0


def hcl(tprior, gmean, gvar, tcount, s1):
    "Assumes a single hyperparameter"
    p0 = np.log(gamma_pdf(tprior, gmean, gvar))
    s0 = len(tcount)
    p0 += s0*(gammaln(tprior*s1)-sum(gammaln(tprior) for i in xrange(s1)))
    for d in xrange(s0):
        for i,t in tcount[d].items():
            p0 += gammaln(t + tprior)
        p0 += (s1-len(tcount[d]))*gammaln(tprior)
        p0 -= gammaln(sum(tcount[d]) + s1*tprior)
    return p0

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
            elif w not in stoplist:
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
    def __init__(self, dirs, npt):
        self.all_words = []
        self.reverse_map = {}
        self.doc_map = {}
        self.documents = []
        self.Ndocuments = 0
        self.Nwords = 0
        self.npt = npt
        self.load_docs(dirs)
        self.Nwords = len(self.all_words)
        self.Ndocuments = len(self.documents)
        self.assignments = [[0 for w in d] for d in self.documents]
        self.mcsample = [[collections.defaultdict(int) for w in d] 
                         for d in self.documents]
        self.initialize()
    
    def load_doc(self,doc):
        "Creates a bag of words for a single document"
        v = []
        document = doc
        for w in get_words(document):
            w = w.lower()
            if not w in self.reverse_map:
                self.reverse_map[w] = self.Nwords
                self.all_words.append(w)
                self.Nwords += 1
            v.append(self.reverse_map[w])
        if not v: return
        self.doc_map[document] = len(self.documents)
        self.docs.append(doc)
        self.documents.append(v)
        
    def load_docs(self, dirs):
        self.Nproducts = 0
        self.prod = []
        self.docs = []
        for id,d in enumerate(dirs):
            for fname in os.listdir(d):
                doc = file(os.path.join(d,fname)).read()
                self.prod.append(id)
                self.load_doc(doc)
            self.Nproducts += 1


    def initialize(self):
        self.generic = collections.defaultdict(lambda:0)
        self.ngeneric = [0]
        self.products = [[collections.defaultdict(lambda:0)
                          for p in xrange(self.Nproducts)] 
                         for i in xrange(self.npt)]
        self.pnames = [["Categoria %s, topico %s"%(p,i)
                          for p in xrange(self.Nproducts)] 
                         for i in xrange(self.npt)]
        self.nproducts = [[[0] for p in xrange(self.Nproducts)] 
                          for i in xrange(self.npt)]
        self.reviews = [collections.defaultdict(lambda: 0) for d in self.documents]
        self.rnames = ["Documento %s"%d for d in xrange(len(self.documents))]
        self.nreviews = [[0] for d in self.documents]
        self.tcount = np.zeros((self.Ndocuments, self.npt+2))
        self.tsamp = np.zeros((self.Ndocuments, self.npt+2))
        self.tprior = np.array([2. for i in xrange(self.npt+2)])
        self.beta = [0.1] + [0.1 for i in xrange(self.npt)] + [.01]
        self.all_topics = [[self.generic]] + self.products+[self.reviews]
        self.all_tnames = [["generic"]] + self.pnames + [self.rnames]
        self.dtopics = []
        self.dsums = []
        tc = self.tprior/sum(self.tprior)
        for d in xrange(self.Ndocuments):
            self.dtopics.append([self.generic]+
                                [self.products[i][self.prod[d]] 
                                 for i in xrange(self.npt)] +
                                [self.reviews[d]] )
            self.dsums.append([self.ngeneric] +
                              [self.nproducts[i][self.prod[d]]
                               for i in xrange(self.npt)] + 
                              [self.nreviews[d]])
            for i,w in enumerate(self.documents[d]):
                t = discrete(tc)
                self.assignments[d][i] = t
                self.tcount[d,t] += 1
                self.dtopics[d][t][w] += 1
                self.dsums[d][t][0] += 1

    def resample_word(self, d,i,w,pt):
        "Resamples the topic assignments of a word"
        to = self.assignments[d][i]
        assert self.documents[d][i] == w, "%s %s %s"%(d,i,w)
        assert self.dtopics[d][to][w] > 0, "%d %d %d %s"%(d, to, w, i)
        assert self.dsums[d][to][0] > 0
        self.dtopics[d][to][w] -= 1
        self.dsums[d][to][0] -= 1
        self.tcount[d,to] -= 1
        pt *= 0
        tc = self.tcount[d] + self.tprior
        for j in xrange(len(pt)):
            pt[j] = (tc[j]*(self.dtopics[d][j][w]+self.beta[j])/
                     (self.dsums[d][j][0]+self.beta[j]*self.Nwords))
        pt /= np.sum(pt)
        nt = discrete(pt)
        self.assignments[d][i] = nt
        self.dtopics[d][nt][w] += 1
        self.dsums[d][nt][0] += 1
        self.tcount[d,nt] += 1


    def resample_tprior(self):
        for i, t in enumerate(self.tprior):
            def partial_lik(t0):
                self.tprior[i] = t0
                return cl(self.tprior, 0.1, 10, self.tcount)
            self.tprior[i] = slice_sample(partial_lik, t)

    def resample_beta(self):
        for i,t in enumerate(self.beta):
            def beta_lik(beta):
                self.beta[i] = beta
                return hcl(self.beta[i],0.1,10,self.all_topics[i],self.Nwords)
            self.beta[i] = slice_sample(beta_lik, self.beta[i])


    def likelihood(self):
        l0 = cl(self.tprior, 0.1, 10, self.tcount)
        for b,t in zip(self.beta,self.all_topics):
            l0 += hcl(b,0.1,10,t,self.Nwords)
        return l0

    def iterate(self):
        pt = np.zeros(len(self.tprior))
        self.resample_tprior()
        self.resample_beta()
        print self.tprior, self.beta, self.likelihood()
        for document in xrange(self.Ndocuments):
            for i,word in enumerate(self.documents[document]):
                self.resample_word(document,i,word,pt)
                self.mcsample[document][i][self.assignments[document][i]] += 1
        self.tsamp += self.tcount
        sys.stdout.flush()

    def print_document(self, d):
        ass = [dargmax(a) for a in self.mcsample[d]]
        txt = self.docs[d]
        i = 0
        s = "" 
        for t,w in get_word_stop(txt):
            if t and 0 < ass[i] < len(self.tprior)-1: 
                s += "\\textbf{"+w+"}"
            else:
                s += w
            i += t
        print "Documento", d, "categoria", self.prod[d]
        print
        print s
        print
        print
               

    def run(self,its):
        "The sampler itself."
        iteration = 0
        print "iterating.."
        for i in xrange(its):
            iteration += 1
            self.iterate()
        for n,i in zip(self.all_tnames[:-1], self.all_topics[:-1]):
            for nn,t in zip(n,i):
                print nn
                print_keyw_topic(self, t, 40)
        for d in xrange(self.Ndocuments):
            self.print_document(d)
            
    

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
    s = LDASampler(sys.argv[1:], NTOPICS)
    s.run(15)

