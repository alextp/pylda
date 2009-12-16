# coding: utf-8

"""
A bayesian factored language model.

See the paper for the model.
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

class TrigramSampler(object):
    def __init__(self,alpha,beta,gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.all_words = []
        self.reverse_map = {}
        self.Ncontexts = 0
        self.all_contexts = []
        self.cont_reverse_map = {}
        self.Nwords = 0
        self.Nfactors = 0
        self.words = []
        self.f1contexts = []
        self.f2contexts = []
        self.ncontexts = []
        self.f1 = []
        self.f2 = []
        self.n = []
        
    def add_crm(self, context):
        if not context in self.cont_reverse_map:
            self.cont_reverse_map[context] = self.Ncontexts
            self.all_contexts.append(f1context)
            self.Ncontexts += 1

    def load_document(self,document)
        for w in get_words(document):
            w = w.lower()
            if not w in self.reverse_map:
                self.reverse_map[w] = self.Nwords
                self.all_words.append(w)
                self.Nwords += 1
            self.words.append(self.reverse_map[w])
            l = self.Nwords
            f1context = tuple(self.words[l-2:l])
            f2context = tuple(self.words[l-3:l-2]+self.words[l-1:l])
            ncontext = tuple(self.words[l-3:l-1])
            [self.add_crm(x) for x in (f1context,f2context,ncontext)]
            self.f1contexts.append(self.cont_reverse_map[f1context])
            self.f2contexts.append(self.cont_reverse_map[f2context])
            self.ncontexts.append(self.cont_reverse_map[ncontext])
            

    def resample_f(self,i, f,c, Cfw, Cw, Cfn, Cn):
        """P(f_w = f) = \frac{C_{fw} + \alpha}{C_{-w}+F\alpha} \frac{C_{cfn} +
  \beta}{C_{c-n}+F\beta} """
        old_class = f[i]
        context = c[i]
        word = self.words[i]
        Cfw[word,old_class] -= 1
        Cw[word] -= 1
        Cfn[context,old_class] -= 1
        Cn[context] -= 1
        aa = Cfw[word] + self.alpha
        bb = Cw[word] + self.Nfactors*self.alpha
        cc = Cfn[context] + self.beta
        dd = Cn[context] + self.Nfactors*self.beta
        pt = (aa/bb)*(cc/dd)
        pt /= np.sum(pt)
        new_class = categorical2(pt)
        f[i] = new_class
        Cfw[word,new_class] += 1
        Cw[word] += 1
        Cfn[context,new_class] += 1
        Cn[context] += 1
        
    def phi_theta_eta(self):
        phi = self.beta*np.ones((self.Nwords,self.Nfactors))
        theta = self.alpha*np.ones((self.Nfactors,self.Nfactors,self.Nfactors))
        eta = self.alpha*np.ones((self.Nfactors,self.Nwords))
        for d in xrange(self.Ndocuments):
            for i,w in enumerate(self.documents[d]):
                t = self.assignments[d][i]
                p[t,w] += 1
                th[d,t] += 1
        return p,th


    def likelihood(self):
        "Computes the likelihood of the parameters"
        f1 = self.Ndocuments*(gammaln(self.Ntopics*self.alpha)-
                              self.Ntopics*gammaln(self.alpha))
        f1 += np.log(gamma_pdf(self.alpha,0.1,1))
        f1 += np.log(gamma_pdf(self.beta,0.1,1))
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

    def iterate(self):
        for document in xrange(self.Ndocuments):
            for i,word in enumerate(self.documents[document]):
                pp = self.cond_dist(document,i,word)

    def resample_alpha(self, lik):
        oldalpha = self.alpha
        self.alpha = np.random.exponential(self.alpha)
        self.pa = self.alpha*self.Nwords
        self.pb = self.beta*self.Ntopics
        nlik = self.likelihood()
        pratio = np.exp(nlik - lik)
        qratio = exp_pdf(self.alpha,oldalpha)/exp_pdf(oldalpha,self.alpha)
        if random.random() < pratio*qratio:
            return nlik
        self.alpha = oldalpha
        self.pa = self.alpha*self.Nwords
        self.pb = self.beta*self.Ntopics
        return lik

    def resample_beta(self, lik):
        oldbeta = self.beta
        self.beta = np.random.exponential(self.beta)
        self.pa = self.alpha*self.Nwords
        self.pb = self.beta*self.Ntopics
        nlik = self.likelihood()
        pratio = np.exp(nlik - lik)
        qratio = exp_pdf(self.beta,oldbeta)/exp_pdf(oldbeta,self.beta)
        if random.random() < pratio*qratio:
            return nlik
        self.beta = oldbeta
        self.pa = self.alpha*self.Nwords
        self.pb = self.beta*self.Ntopics
        return lik

    def run(self,Ntopics,burnin, interval,nsamples):
        "The sampler itself."
        self.Ntopics = Ntopics
        self.Nwords = len(self.all_words)
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
            self.iterate()
            lik = self.likelihood()
            lik = self.resample_alpha(lik)
            lik = self.resample_beta(lik)
            print self.alpha,self.beta,lik
            self.print_topic_proportions()
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

    def parse_lda_data(self,prefix):
        data_f = file(prefix+".data")
        vocab = [a.strip() for a in file(prefix+".vocab")]
        self.all_words = vocab
        self.make_reverse_map()
        self.Nwords = len(vocab)
        data = [a.strip().split() for a in data_f]
        self.Ndocuments = len(data)
        self.documents = [[] for i in xrange(self.Ndocuments)]
        for doc in xrange(self.Ndocuments):
            for word in data[doc][1:]:
                w,c = map(int,word.split(":"))
                if w >= len(self.all_words):
                    print w,c
                    w = len(all_words)-1
                [self.documents[doc].append(w) for i in xrange(c)]
    

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


def most_likely_topic_proportions(phi,example):
    p = phi.copy().T
    t = np.zeros(len(p[0]))
    for i,w in enumerate(example):
        if w > 0.01:
            t += w*p[i]
    return t
        
def best_good_words(tprop,phi,good_words,vocab,n):
    #print tprop/sum(tprop)
    rmap = dict((w,i) for i,w in enumerate(vocab))
    good_index = [rmap[i] for i in good_words]
    wprops = sum(phi[i]*tprop[i] for i in xrange(len(tprop)))
    mean_words = sum(phi)/10.
    #wprops -= mean_words
    #print [wprops[i] for i in good_index]
    rtops = np.array([wprops[i] for i in good_index])
    best = np.argsort(rtops)
    return set(vocab[good_index[i]] for i in best[:n])
    

def compute_likely_words_set(phi,example,vocab,n):
    good_words = [w for w in vocab if w.replace("x",'').replace("O",'')]
    m = most_likely_topic_proportions(phi, example)
    return best_good_words(m, phi, good_words, vocab,n)
    

def recall(examples,vocab,phi,good_f,n):
    good_words = [l.strip() for l in file(good_f)]
    rec = 0.
    for s,g in zip([compute_likely_words_set(phi,example,vocab,n) for example in examples],good_words):
        if g in s:
            rec += 1
    return rec/len(good_words)

def split_train_test(basename, fraction):
    vocab = [w.strip() for w in file(basename+".vocab")]
    reverse_map = dict((w,i) for i,w in enumerate(vocab))
    good_indexes = [i for i,w in enumerate(vocab) 
                    if w.replace("x",'').replace("O",'')]
    good_set = set(good_indexes)
    data = [bag.strip() for bag in file(basename+".data")]
    top_train = int(len(data)*fraction)
    train = data[:top_train]
    file("training.data","w").write("\n".join(train))
    test = data[top_train:]
    new_test = [' '.join([b for b in bag.strip().split()
                          if not int(b.split(":")[0]) in good_set])
                for bag in test]
    good_words = [' '.join([vocab[int(b.split(':')[0])] for b in bag.strip().split()
                          if int(b.split(":")[0]) in good_set])
                for bag in test]
    file("good.words","w").write("\n".join(good_words))
    file("test.data","w").write("\n".join(new_test))
    

def parse_bag(bag, Nwords):
    b = np.zeros(Nwords)
    for bags in bag.split()[1:]:
        w,c = map(int,bags.split(":"))
        b[w] += c
    return b

def load_blei_phi(f):
    data = [a.split() for a in file(f).read().split('\n') if a.strip()]
    d = np.zeros((len(data),len(data[0])))
    for i in xrange(len(data)):
        for j in xrange(len(data[0])):
            try:
                d[i,j] = float(data[i][j])
            except:
                print i,j,len(data[i]),d.shape,len(data)
    return d
    
        
f = ("/home/top/textos/Douglas Adams/Douglas Adams -"
     " So Long, and Thanks For All the Fish.txt")

if __name__=='__main__':
    s = LDASampler()
    [s.load_as_bag(x) for x in file(f).read().split("\r\n\r\n")]
    phi,theta = s.run(10, 20, 3, 5)
    print "returned"
    s.print_topics(phi,10)
    
    
