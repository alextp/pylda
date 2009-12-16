# coding: utf-8

"""
Semi-superised LDA
"""

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

def categorical2(probs):
    return np.argmax(np.random.multinomial(1,probs))

def mean(x):
    return sum(x)/len(x)


def gamma_pdf(x,k,theta):
    x,k,theta = map(float,(x,k,theta))
    return (x**(k-1))*(math.exp(-x/theta))/((theta**k)*gamma(k))


def exp_pdf(x,k):
    return k*math.exp(-k*x)

class LDASampler(object):
    def __init__(self):
        self.all_words = []
        self.reverse_map = {}
        self.all_topics = []
        self.treverse_map = {}
        self.documents = []
        self.Ndocuments = 0
        self.Nwords = 0
        self.alpha = 0.01
        self.beta = 0.001
    
    def phi_theta(self):
        p = self.beta*np.ones((self.Ntopics,self.Nwords)) 
        th = self.alpha*np.ones((self.Ndocuments,self.Ntopics))
        for d in xrange(self.Ndocuments):
            for i,w in enumerate(self.documents[d]):
                t = self.assignments[d][i]
                p[t,w] += 1
                th[d,t] += 1
        return p,th

    def load_data(self, bp, documents_file, topics_file, vocab_file):
        self.documents = []
        for line in file(bp+documents_file):
            line = line.split()[1:]
            doc = []
            for word in line:
                w,c = map(int,word.split(":"))
                if w >= self.Nwords:
                    self.Nwords = w+1
                for i in xrange(c):
                    doc.append(w)
            self.documents.append(doc)
        for line in file(bp+vocab_file):
            line = line.strip()
            self.reverse_map[line] = len(self.all_words)
            self.all_words.append(line)
        self.topics = []
        for line in file(bp+topics_file):
            doc = []
            for topic in line.split():
                topic = topic.strip()
                if not topic in self.treverse_map:
                    self.treverse_map[topic] = len(self.all_topics)
                    self.all_topics.append(topic)
                doc.append(self.treverse_map[topic])
            self.topics.append(doc)
        self.treverse_map["NULL"] = len(self.all_topics)
        self.all_topics.append("NULL")
        self.topic_indicators = []
        for doc in self.topics:
            indicator = np.zeros(len(self.all_topics))
            for t in doc:
                indicator[t] = 1
            indicator[-1] = 1
            self.topic_indicators.append(indicator)

    def likelihood(self):
        "Computes the likelihood of the parameters"
        f1 = self.Ndocuments*(gammaln(self.Ntopics*self.alpha)-
                              self.Ntopics*gammaln(self.alpha))
        f1 *= gamma_pdf(self.alpha,1,1)
        f1 *= gamma_pdf(self.beta,1,1)
        vt = np.zeros(self.Ntopics)
        f2 = 0.
        for d in xrange(self.Ndocuments):
            vt.fill(0)
            for i,w in enumerate(self.documents[d]):
                vt[self.assignments[d][i]] += 1
            vt += self.alpha
            f2t1 = np.sum(gammaln(vt))
            f2t2 = gammaln(self.Ntdcs[d]+self.Ntopics*self.alpha)
            f2 += f2t1-f2t2
        return f1 + f2

    def initialize(self):
        for d in xrange(self.Ndocuments):
            for i,w in enumerate(self.documents[d]):
                t = random.randint(0,self.Ntopics-1)
                self.assignments[d][i] = t
                self.Nwt[w,t] += 1
                self.Ntd[d,t] += 1
                self.Nwtcs[t] += 1
                self.Ntdcs[d] += 1
        self.pa = self.alpha*self.Nwords
        self.pb = self.beta*self.Ntopics

    def cond_dist(self, d,i,w,f):
        to = self.assignments[d][i]
        self.Nwtcs[to] -= 1
        self.Ntdcs[d] -= 1
        self.Nwt[w,to] -= 1
        self.Ntd[d,to] -= 1
        aa = (self.Nwt[w]+self.beta)
        bb = (self.Nwtcs+self.pb)
        cc = (self.Ntd[d]+self.alpha)
        dd = (self.Ntdcs[d]+self.pa)
        pt = (aa/bb)*(cc/dd)
        if float(d)/self.Ndocuments < f:
            pt *= self.topic_indicators[d]
        pt /= np.sum(pt)
        nt = categorical2(pt)
        self.assignments[d][i] = nt
        self.Nwtcs[nt] += 1
        self.Ntdcs[d] += 1
        self.Nwt[w,nt] += 1
        self.Ntd[d,nt] += 1
        return pt[nt]

    def iterate(self,fraction):
        for document in xrange(self.Ndocuments):
            for i,word in enumerate(self.documents[document]):
                pp = self.cond_dist(document,i,word,fraction)

    def run(self,burnin,interval,nsamples,fraction):
        "The sampler itself."
        self.Ntopics = len(self.all_topics)
        #self.Nwords = len(self.all_words)
        self.Ndocuments = len(self.documents)
        self.assignments = [[0 for w in d] for d in self.documents]
        self.Nwt = np.zeros((self.Nwords,self.Ntopics))
        self.Ntd = np.zeros((self.Ndocuments,self.Ntopics))
        self.Nwtcs = np.zeros(self.Ntopics)
        self.Ntdcs = np.zeros(self.Ndocuments)
        old_lik = -np.inf
        samples = []
        self.initialize()
        iteration = 0
        while len(samples) <  nsamples:
            iteration += 1
            self.iterate(fraction)
            lik = self.likelihood()
            #self.print_topic_proportions()
            print lik
            if iteration > burnin and iteration % interval == 0:
                samples.append(self.phi_theta())
        return mean([a[0] for a in samples]), mean([a[1] for a in samples])

    def print_topic_proportions(self):
        tcounts = np.zeros(self.Ntopics)
        for d in xrange(self.Ndocuments):
            for w in self.assignments[d]:
                tcounts[w] += 1
        tcounts /= sum(tcounts)
        for t in tcounts:
            print "%.3f"%t,
        print

    def print_topic(self,phi, t, n):
        print "topico", t,":"
        s = np.argsort(-phi[t])
        for w in s[:n]:
            print "     ",self.all_words[w]


    def print_topics(self,phi,n):
        for t in xrange(len(phi)):
            self.print_topic(phi,t,n)
            print

    def make_reverse_map(self):
        for i,w in enumerate(self.all_words):
            self.reverse_map[w] = i

    

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

    

def parse_bag(bag, Nwords):
    b = np.zeros(Nwords)
    for bags in bag.split()[1:]:
        w,c = map(int,bags.split(":"))
        b[w] += c
    return b
    
        
f = ("/home/top/textos/Douglas Adams/Douglas Adams -"
     " So Long, and Thanks For All the Fish.txt")

if __name__=='__main__':
    bp = ""
    s = LDASampler()
    s.load_data(bp,"boston-training.data","boston-test.good","boston-training.vocab")
    phi, theta = s.run(100,5,10,0.8)
    ndocs = s.Ndocuments
    for i in xrange(int(0.8*ndocs), ndocs):
        pi = theta[i]
        for word in np.argsort(-pi)[:5]:
            print s.all_topics[word],
        print "|",
        for w in s.topics[i]:
            print s.all_topics[w],
        print
    
    
