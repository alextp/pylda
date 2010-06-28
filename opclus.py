# coding: utf-8

"""
Semi-superised LDA
"""

import numpy as np
import random
import math
from scipy.special import gamma,gammaln
from scipy import weave
import sys, os


stop = set(l.strip() for l in file("/home/top/downloads/multi-task-review/sorted_data/stopwords"))

import re
wre = re.compile(r"(\w)+")
def get_words(text):
    "A simple tokenizer"
    l = 0
    while l < len(text):
        s = wre.search(text,l)
        try:
            st = text[s.start():s.end()].lower()
            if not st in stop:
                yield st 
            l = s.end()
        except:
            break

def bigrams(it):
    prev = None
    for i in it:
        if prev:
            yield prev,i
        prev = i

def categorical2(probs):
    return np.argmax(np.random.multinomial(1,probs))


def parse_file(reviews, bp, dire, p, l, fname):
    f = os.path.join(bp,dire,fname)
    in_review = False
    text = ""
    for line in file(f):
        if in_review:
            if line.strip() == "</review_text>":
                in_review = False
                reviews.append((p, l, text))
                text = ""
            else:
                text += " " + line
        else:
            if line.strip() == "<review_text>":
                in_review = True
    return reviews

def parse_reviews(bp):
    reviews = []
    for p,dire in enumerate(os.listdir(bp)):
        if not "." in dire and not "stopwords" in dire:
            parse_file(reviews, bp, dire, p, "n", "negative.review")
            parse_file(reviews, bp, dire, p, "p", "positive.review")
            parse_file(reviews, bp, dire, p, "u", "unlabeled.review")
    return reviews
    


class OpinionSampler(object):
    def __init__(self, reviews, nops):
        print "init"
        random.shuffle(reviews)
        reviews = [r for r in reviews if r[1] != "u"]
        self.product = [r[0] for r in reviews]
        self.all_products = list(sorted(set(self.product)))
        self.mprod = max(self.all_products)+1
        self.label = [r[1] for r in reviews]
        self.text = [r[2] for r in reviews]
        print "init 2"
        self.docs = []
        self.reverse_map = {}
        self.all_words = []
        for t in self.text:
            doc = []
            for w in bigrams(get_words(t)):
                if not w in self.reverse_map:
                    self.reverse_map[w] = len(self.all_words)
                    self.all_words.append(w)
                doc.append(self.reverse_map[w])
            self.docs.append(doc)
        print "init 3"
        self.Ndocuments = len(self.docs)
        self.Nwords = len(self.all_words)
        self.Nops = nops
        self.alpha = 1.
        self.beta = 100.
        self.op_counts = np.zeros(nops)+self.beta
        self.ops = np.array([np.zeros(len(self.all_words))+self.alpha 
                             for i in xrange(nops)])
        self.sops = np.array([np.sum(s) for s in self.ops])
        self.prods = [np.zeros(len(self.all_words))+self.alpha for i in xrange(self.mprod)]
        self.sprods = [np.sum(s) for s in self.prods]
        self.generic = np.zeros(len(self.all_words))+self.alpha
        self.sgen = np.sum(self.generic)
        self.initialize()


    def initialize(self):
        print "init 4"
        self.assign_ops = [random.randint(0, len(self.ops)-1) for i in self.docs]
        #d= {"p": 1, "n":0, "u": 2}
        #self.assign_ops = [d[i] for i in self.label]
        self.assign_words = []
        print "init 5"
        ps = np.array([1., 3., 1.])
        ps /= np.sum(ps)
        for d in xrange(self.Ndocuments):
            ass = []
            self.op_counts[self.assign_ops[d]] += 1
            rel = 0
            for i,w in enumerate(self.docs[d]):
                t = categorical2(ps)
                ass.append(t)
                if t == 1:
                    self.ops[self.assign_ops[d]][w] += 1
                    self.sops[self.assign_ops[d]] += 1
                    rel += 1
                elif t == 0:
                    self.prods[self.product[d]][w] += 1
                    self.sprods[self.product[d]] += 1                    
                else:
                    self.generic[w] += 1
                    self.sgen += 1
            self.assign_words.append(ass)
        print "init 6"

    def w_cond_dist(self, d,w):
        op = self.ops[self.assign_ops[d]]
        sop = self.sops[self.assign_ops[d]]
        prod = self.prods[self.product[d]]
        sprod = self.sprods[self.product[d]]
        generic = self.generic
        sgen = self.sgen
        ww = self.docs[d][w]
        if self.assign_words[d][w] == 1:
            op[ww] -= 1
            sop -= 1
        elif self.assign_words[d][w] == 0:
            prod[ww] -= 1
            sprod -= 1
        else:
            generic[ww] -= 1
            sgen -= 1
        ps = np.zeros(3)
        ps[1] = (op[ww])/((sop))
        ps[0] = (prod[ww])/((sprod))
        ps[2] = 0 #(generic[ww])/((sgen))
        ps /= np.sum(ps)
        t = categorical2(ps)
        self.assign_words[d][w] = t
        if self.assign_words[d][w] == 1:
            op[ww] += 1
            sop += 1
        elif self.assign_words[d][w] == 0:
            prod[ww] += 1
            sprod += 1
        else:
            generic[ww] += 1
            sgen += 1

    def rel_words(self, d):
        rwd = []
        t = 0
        for i,w in enumerate(self.assign_words[d]):
            if w == 1:
                rwd.append(self.docs[d][i])
                t += 1
        return rwd, t

    def c_cond_dist(self, d):
        rwd, t = self.rel_words(d)
        for w in rwd:
            self.ops[self.assign_ops[d]][w] -= 1
        self.sops[self.assign_ops[d]] -= t
        self.op_counts[self.assign_ops[d]] -= 1

        if t == 0:
            nop = random.randint(0,len(self.ops)-1)
            self.op_counts[nop] += 1
            self.assign_ops[d] = nop
            return
        ps = np.zeros(len(self.ops))
        for i in xrange(len(ps)):
            for w in rwd:
                ps[i] += np.log((self.ops[i][w])/self.sops[i])
        ps = np.exp(ps)
        ps /= np.sum(ps)
        nop = categorical2(ps)
        self.assign_ops[d] = nop
        for w in rwd:
            self.ops[nop][w] += 1
        self.sops[nop] += t
        self.op_counts[nop] += 1


    def old_c_cond_dist(self, d):
        rwd, t = self.rel_words(d)
            
        self.ops[self.assign_ops[d]] -= rwd
        self.sops[self.assign_ops[d]] -= t
        self.op_counts[self.assign_ops[d]] -= 1

        if t == 0:
            nop = random.randint(0,len(self.ops)-1)
            self.op_counts[nop] += 1
            self.assign_ops[d] = nop
            return
        ps = np.sum(rwd*np.log((self.ops / self.sops.reshape((-1,1)))), axis=1)
        ps = np.exp(ps)
        ps /= np.sum(ps)
        nop = categorical2(ps)
        self.assign_ops[d] = nop
        self.ops[nop] += rwd
        self.sops[nop] += t
        self.op_counts[nop] += 1
        

    def iterate(self, it):
        for document in xrange(self.Ndocuments):
            if document % 1000 == 0:
                pass #print "document", document, self.Ndocuments
            self.c_cond_dist(document)
            for i in xrange(len(self.docs[document])):
                self.w_cond_dist(document, i)

    def likelihood(self):
        lik = 0
        for d in xrange(self.Ndocuments):
            for i,w in enumerate(self.docs[d]):
                if self.assign_words[d][i] == 1:
                    ps = self.ops[self.assign_ops[d]]
                    sps = self.sops[self.assign_ops[d]]
                elif self.assign_words[d][i] == 0:
                    ps = self.prods[self.product[d]]
                    sps = self.sprods[self.product[d]]
                else:
                    ps = self.generic
                    sps = self.sgen
                lik += np.log((ps[w]-1)/(sps-1))
                if lik != lik:
                    print "nan, shit"
                    print str(ps), ps[w]+self.alpha/np.sum(ps+self.alpha)
                    return 0.
        return lik

    def run(self,nsamples):
        "The sampler itself."
        old_lik = -np.inf
        iteration = 0
        self.print_op_proportions()
        for i in xrange(nsamples):
            iteration += 1
            self.iterate(i)
            self.print_op_proportions()
            self.print_prod_proportions()
            lik = self.likelihood()
            #self.print_topic_proportions()
            print lik

    def print_op_proportions(self):
        props = [{"n":0, "p":0, "u":0} for o in self.ops]
        for d in xrange(len(self.docs)):
            props[self.assign_ops[d]][self.label[d]] += 1
        p2 = []
        for i,p in enumerate(props):
            print
            print "op", i, self.op_counts[i]/np.sum(self.op_counts)
            ps = self.ops[i]+self.alpha
            norm = np.sum(ps)
            top_k = np.argsort(-ps)[:30]
            for t in top_k:
                print self.all_words[t], ps[t]/float(norm)

        print
        print "opc", 
        for i,p in enumerate(props):
            c_p = p["p"]
            c_n = p["n"]
            c_u = p["u"]
            c_t = float(c_p+c_n+c_u)
            if c_t == 0: continue
            if c_n+c_p == 0: continue
            print "%5f" %(c_p/float(c_n+c_p)),
        print


    def print_prod_proportions(self):
        p2 = []
        cr = np.zeros(len(self.prods))
        cp = np.zeros(len(self.prods))
        cg = np.zeros(len(self.prods))
        cc = [cp,cr,cg]
        for i in xrange(len(self.docs)):
            for j in xrange(len(self.docs[i])):
                cc[self.assign_words[i][j]][self.product[i]] += 1
        for i,p in enumerate(self.prods):
            ct = cp[i] + cr[i] + cg[i]
            print "prod", i, cp[i]/ct, cr[i]/ct, cg[i]/ct
        self.print_prod(0, self.generic, "generic", 0, 0)

    def print_prod(self, i, p, pr, cpi, cgi):
        ps = p+self.alpha
        norm = np.sum(ps)
        top_k = np.argsort(-ps)[:20]
        print
        print pr,i, cpi, cgi
        for t in top_k:
            print self.all_words[t], ps[t]/float(norm)


    


if __name__=='__main__':
    pass
    
