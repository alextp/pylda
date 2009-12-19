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
from scipy.weave import converters
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
        self.Nwords = 0
        self.Nfactors = 0
        self.words = []
        self.trigrams = []
        self.Ntrigrams = 0
        
    def load_document(self,document):
        self.words, self.trigrams = self.read_document(document)
        self.Ntrigrams = len(self.trigrams)
    
    def read_document(self,document, update=True):
        words = []
        trigrams = []
        for w in get_words(document):
            w = w.lower()
            if not w in self.reverse_map:
                if not update: continue
                self.reverse_map[w] = self.Nwords
                self.all_words.append(w)
                self.Nwords += 1
            words.append(self.reverse_map[w])
            tg = words[-3:]
            if len(tg) == 3:
                trigrams.append(tg)
        return words, trigrams

    def resample_f(self,ass,i, f, c, Cfw, Cw, Cfn, Cn):
        """P(f_w = f) = \frac{C_{fw} + \alpha}{C_{-w}+F\alpha} \frac{C_{cfn} +
  \beta}{C_{c-n}+F\beta} """
        old_class = f[ass,i]
        cn = tuple([(c[a] if a != i else self.Nfactors) for a in xrange(len(c))])
        word = self.trigrams[ass][i]
        Cfw[word,old_class] -= 1
        assert Cfw[word,old_class] >= 0
        Cw[word] -= 1
        Cfn[c] -= 1
        Cn[cn] -= 1
        aa = Cfw[word] + self.alpha
        bb = Cw[word] + self.Nfactors*self.alpha
        cc = Cfn[c] + self.beta
        dd = Cn[cn] + self.Nfactors*self.beta
        pt = (aa/bb)*(cc/dd)
        pt /= np.sum(pt)
        new_class = categorical2(pt)
        f[ass,i] = new_class
        Cfw[word,new_class] += 1
        Cw[word] += 1
        Cfn[c,new_class] += 1
        Cn[cn] += 1

    def resample_trigram(self,i):
        assignment = self.assignments[i]
        c = tuple(assignment)
        self.resample_f(i,0, self.assignments, c, self.Cfw, self.Cw, self.Cfn, self.Cn)
        assignment = self.assignments[i]
        c = tuple(assignment)
        self.resample_f(i,1, self.assignments, c, self.Cfw, self.Cw, self.Cfn, self.Cn)
        assignment = self.assignments[i]
        c = tuple(assignment)
        self.resample_f(i,2, self.assignments, c, self.Cfw, self.Cw, self.Cfn, self.Cn)
        
    def initialize(self):
        self.assignments = np.zeros((self.Ntrigrams, 3))
        self.Cfw = np.zeros((self.Nwords, self.Nfactors))
        self.Cw = np.zeros((self.Nwords))
        self.Cfn = np.zeros((self.Nfactors+1, self.Nfactors+1, self.Nfactors+1))
        self.Cn = np.zeros((self.Nfactors+1, self.Nfactors+1, self.Nfactors+1))
        for i in xrange(len(self.assignments)):
            for j in xrange(3):
                a = np.random.randint(0, self.Nfactors)
                self.assignments[i,j] = a
                self.Cfw[self.trigrams[i][j],a] += 1
                self.Cw[self.trigrams[i][j]] += 1
            c = tuple(self.assignments[i])
            self.Cfn[c] += 1
            for j in xrange(3):
                cc = tuple(list(c[:j])+[self.Nfactors]+list(c[j+1:]))
                self.Cn[cc] += 1

    def iterate(self):
        for tg in xrange(self.Ntrigrams):
            self.resample_trigram(tg)

    def run(self,Nfactors,burnin, interval, nsamples, ll):
        "The sampler itself."
        old_lik = -np.inf
        self.Nfactors = Nfactors
        samples = []
        self.initialize()
        iteration = 0
        phi, theta = self.phi_theta_eta()
        lik = self.likelihood(self.trigrams, phi, theta)
        ll[iteration] = lik
        while len(samples) <  nsamples:
            iteration += 1
            self.iterate()
            phi, theta = self.phi_theta_eta()
            lik = self.likelihood(self.trigrams, phi, theta)
            #print lik
            ll[iteration] = lik
            if iteration > burnin and iteration % interval == 0:
                samples.append((phi,theta))
        return mean([a[0] for a in samples]), mean([a[1] for a in samples])

    def phi_theta_eta(self):
        phi = self.alpha*np.ones((self.Nfactors,self.Nfactors,self.Nfactors))
        theta = self.beta*np.ones((self.Nwords,self.Nfactors))
        for i in xrange(len(self.trigrams)):
            for w,a in zip(self.trigrams[i],self.assignments[i]):
                theta[w,a] += 1
            c = tuple(self.assignments[i])
            phi[c] += 1
        for i in xrange(len(theta)):
            theta[i] /= sum(theta[i])
        for i in xrange(phi.shape[0]):
            for j in xrange(phi.shape[1]):
                phi[i,j] /= np.sum(phi[i,j])
        return phi, theta
    
    def prob(self, t, phi, theta, pprob):
        """  P(c|ab) =  \sum_{f_1,f_2,n} P(c|n)P(f_1|a)P(f_2|b)P(n|f_1,f_2) """
        a = theta[t[0]]
        b = theta[t[1]]
        c = theta[t[2]]
        #return np.dot(a,np.dot(b,np.dot(c,phi)))
        sum = 0
        for f1 in xrange(self.Nfactors):
            for f2 in xrange(self.Nfactors):
                for f3 in xrange(self.Nfactors):
                    factor = a[f1]*b[f2]*phi[f1,f2,f3]
                    n = c[f3]*(1./self.Nwords)/pprob[f3]
                    sum += n*factor
        return sum


    def weave_prob(self, t, phi, theta, pprob):
        """  P(c|ab) =  \sum_{f_1,f_2,n} P(c|n)P(f_1|a)P(f_2|b)P(n|f_1,f_2) """
        a = theta[t[0]]
        b = theta[t[1]]
        c = theta[t[2]]
        #return np.dot(a,np.dot(b,np.dot(c,phi)))
        Nfactors = self.Nfactors
        Nwords = self.Nwords
        code = """
        double sum = 0;
        for (int f1 = 0; f1 <  Nfactors; ++f1) {
          for (int f2 = 0; f2 <  Nfactors; ++f2) {
            for (int f3 = 0; f3 <  Nfactors; ++f3) {
              double factor = a(f1)*b(f2)*phi(f1,f2,f3);
              double n = c(f3)*(1./Nwords)/pprob(f3);
              sum += n*factor;
            }
          }
        }
        return_val = sum;
        """
        soma = weave.inline(code,["a", "b", "c", "Nfactors", "pprob", "phi", "Nwords"],
                            type_converters=converters.blitz,
                            compiler = 'gcc')
        return soma
                    

    def likelihood(self, trigrams, phi, theta):
        "Computes the likelihood of the parameters"
        loglik = 0
        pprob = sum(theta)/len(theta)
        for i in xrange(len(trigrams)):
            p = self.weave_prob(trigrams[i], phi, theta, pprob)
            loglik += math.log(p)
        return loglik
            
    def held_out_likelihood(self, text, phi, theta):
        words, trigrams = self.read_document(text, update=False)
        return self.likelihood(trigrams,phi,theta)
        



    
        
f = ("/home/top/textos/Douglas Adams/Douglas Adams -"
     " So Long, and Thanks For All the Fish.txt")

f2 = ("/home/top/textos/Douglas Adams/"
      "Adams, Douglas (DG1) Dirk Gently's Holistic Detective Agency.txt")

if __name__=='__main__':
    s = TrigramSampler(0.1, 0.1, 0.1)
    s.load_document(file(f).read())
    ll = np.zeros((20,200))
    a = 0
    import sys
    sys.stdout.flush()
    for i in [5, 10, 15, 20, 25, 30, 40, 50]:
        phi,theta = s.run(i, 100, 10, 3, ll[a])
        a += 1
        print i, s.held_out_likelihood(file(f2).read(), phi, theta)
        sys.stdout.flush()
    for a,i in enumerate(ll.T):
        print a,
        for j in i:
            print j,
        print
    
    
