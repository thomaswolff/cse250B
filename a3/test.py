#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + collapsed Gibbs sampling
# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

import numpy
import string, re, timeit, math

NUMBER_OF_DOCUMENTS = 0
NUMBER_OF_TERMS = 0
NUMBER_OF_NON_ZERO_VALUES = 0


# C is a pseudocount. Should have 0<c<=1 and c<T/M
# Tettas is of length M and can be viewed as probability for word j
# T = Sum of sizes of all training data
# V = 
# M = size of vocabulary
# K = number of topics
# n = length of documents in words

class Document(object):

    count_vector = {}
    category = 0
    
    alphas = []
    betas = []

    words = []
    filename = ""
    length = 0

    def __init__(self, filename, category):
        self.category = category
        self.filename = filename
        self.count_vector = {}

    def getProbability(self, tettas):
        xSum = 0.0
        tettaSum = 0.0
        for key, value in self.count_vector.iteritems():
            print "value: " + str(value)
            print "log x: " + str(math.log(math.factorial(value)))
            xSum += math.log(math.factorial(value))
            tettaSum += value * math.log(tettas[key])

        result = math.log(math.factorial(self.length))
        print "1: " + str(result)
        result -= xSum
        print "2: " + str(xSum)
        result += tettaSum
        print "3: " + str(tettaSum)
        return result




        return 0
    def addCount(self, index, count):
        self.count_vector[index] = count
        self.length += count

    def __str__(self):
        return "Document: " + self.filename + "\n" +"Category: " + self.category + "\n" +"Dict: " + str(self.count_vector)

def findSumOfAllSizes(vocabulary):
    sum = 0
    for key, value in vocabulary.iteritems():
        sum += value[1]
    return sum

def calculateTettas(tettas, vocabulary, c):
    t = findSumOfAllSizes(vocabulary)
    tMarked = (len(vocabulary) * c) + t
    for i in range(len(tettas)):
        allApperances = vocabulary[i+1][1]
        tettas[i] = (1/tMarked) * (c + allApperances)






def readFile(filename):
    with open ("./data/" + filename) as data:
        lines = data.readlines()
    return lines

def readWords(filename):
    lines = readFile(filename)
    words = []
    exclude = set(string.punctuation)
    table = string.maketrans("","")
    for line in lines:
        line = line.lower()
        line = line.rstrip()
        line = line.translate(table, string.punctuation)
        wordsInLine = line.split(' ')
        for word in wordsInLine:
            words.append(word)
    return words

def readTerms():
    lines = readFile("terms_detailed.txt")
    vocabulary = {}
    for line in lines:
        arr = line.rstrip().split(' ')
        vocabulary[int(arr[0])] = (arr[1], int(arr[2]))
    return vocabulary


def readTermMatrix(documents):
    lines = readFile("docbyterm.txt")
    if len(lines) > 0:
        first_line = lines[0].rstrip().split(' ')
        NUMBER_OF_DOCUMENTS = first_line[0]
        NUMBER_OF_TERMS = first_line[1]
        NUMBER_OF_NON_ZERO_VALUES = first_line[2]
    for i in range(1, len(lines)):
        arr = lines[i].rstrip().split(' ')
        if len(arr) > 2:
            doc = documents[int(arr[0]) - 1]
            doc.addCount(int(arr[1]), int(arr[2]))
    print documents[0].count_vector
    print documents[0].filename
    print documents[0].category


def readFileNames():
    raw_lines = readFile("documents.txt")
    documents = []
    for line in raw_lines:
        temp = line.rstrip().split(' ')
        filename = ""
        category = 0
        if (len(temp) > 1):
            filename = temp[0]
            category = temp[1]
        words = readWords("classic/" + filename)
        documents.append(Document(filename, category))
    print documents[0].count_vector
    return documents

class LDA:
    def __init__(self, K, alpha, beta, docs, V, smartinit):
        self.K = K
        self.alpha = alpha # parameter of topics prior
        self.beta = beta   # parameter of words prior
        self.docs = docs
        self.V = V

        self.z_m_n = [] # topics of words of documents
        self.n_m_z = numpy.zeros((len(self.docs), K)) + alpha     # word count of each document and topic
        self.n_z_t = numpy.zeros((K, V)) + beta # word count of each topic and vocabulary
        self.n_z = numpy.zeros(K) + V * beta    # word count of each topic

        self.N = 0
        for m, doc in enumerate(docs):
            self.N += doc.length
            z_n = []
            for term, count in doc.count_vector.iteritems():
                if smartinit:
                    #print "Term: " + str(term)
                    #print "nzt: " + str(self.n_z_t[:, term-1])
                    p_z = self.n_z_t[:, term-1] * self.n_m_z[m] / self.n_z
                    z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
                else:
                    z = numpy.random.randint(0, K)
                z_n.append(z)
                self.n_m_z[m, z] += 1
                self.n_z_t[z, term-1] += 1
                self.n_z[z] += 1
            self.z_m_n.append(numpy.array(z_n))

    def inference(self):
        """learning once iteration"""
        for m, doc in enumerate(self.docs):
            z_n = self.z_m_n[m]
            n_m_z = self.n_m_z[m]
            n = 0
            for term, count in doc.count_vector.iteritems():
                # discount for n-th word t with topic z
                z = z_n[n]
                n_m_z[z] -= 1
                self.n_z_t[z, term-1] -= 1
                self.n_z[z] -= 1

                # sampling topic new_z for t
                p_z = self.n_z_t[:, term-1] * n_m_z / self.n_z
                new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

                # set z the new topic and increment counters
                z_n[n] = new_z
                n_m_z[new_z] += 1
                self.n_z_t[new_z, term-1] += 1
                self.n_z[new_z] += 1
                n+=1

    def worddist(self):
        """get topic-word distribution"""
        return self.n_z_t / self.n_z[:, numpy.newaxis]

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.worddist()
        log_per = 0
        N = 0
        Kalpha = self.K * self.alpha
        for m, doc in enumerate(docs):
            theta = self.n_m_z[m] / (self.docs[m].length + Kalpha)
            for term, count in doc.count_vector.iteritems():
                log_per -= numpy.log(numpy.inner(phi[:,term-1], theta))
            N += doc.length
        return numpy.exp(log_per / N)

def lda_learning(lda, iteration, voca):
    pre_perp = lda.perplexity()
    print "initial perplexity=%f" % pre_perp
    for i in range(iteration):
        lda.inference()
        perp = lda.perplexity()
        print "-%d p=%f" % (i + 1, perp)
        if pre_perp:
            if pre_perp < perp:
                output_word_topic_dist(lda, voca)
                pre_perp = None
            else:
                pre_perp = perp
    output_word_topic_dist(lda, voca)

def output_word_topic_dist(lda, voca):
    zcount = numpy.zeros(lda.K, dtype=int)
    wordcount = [dict() for k in xrange(lda.K)]
    for xlist, zlist in zip(lda.docs, lda.z_m_n):
        for x, z in zip(xlist.count_vector.iteritems(), zlist):
            zcount[z] += 1
            if x in wordcount[z]:
                wordcount[z][x] += 1
            else:
                wordcount[z][x] = 1

    phi = lda.worddist()
    for k in xrange(lda.K):
        print "\n-- topic: %d (%d words)" % (k, zcount[k])
        for w in numpy.argsort(-phi[k])[:20]:
            print "%s: %f (%d)" % (voca[w][0], phi[k,w], wordcount[k].get(w,0))

def main():
    k = 10
    alpha = 0.1
    beta = 0.1

    documents = readFileNames()
    vocabulary = readTerms()
    readTermMatrix(documents)

    #numpy.random.seed(options.seed)

    lda = LDA(k, alpha, beta, documents, len(vocabulary), False)
    print "corpus=%d, K=%d, a=%f, b=%f" % (len(vocabulary), k, alpha, beta)

    #import cProfile
    #cProfile.runctx('lda_learning(lda, options.iteration, voca)', globals(), locals(), 'lda.profile')
    lda_learning(lda, 5, vocabulary)

if __name__ == "__main__":
    main()










