#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + collapsed Gibbs sampling
# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

import numpy
import string, re, timeit, math
import json
from pprint import pprint
import xml.etree.ElementTree as ET
#from nltk import PorterStemmer
from sets import Set
import re, string, timeit
from stemming.porter2 import stem

exclude = set(string.punctuation)
table = string.maketrans("","")
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

    category = 0
    
    alphas = []
    betas = []

    words = []
    filename = ""
    length = 0

    def __init__(self, filename, category, words):
        self.category = category
        self.filename = filename
        self.words = words


    def __str__(self):
        return "Document: " + self.filename + "\n" +"Category: " + self.category + "\n" +"Dict: " + str(self.count_vector)

def findSumOfAllSizes(vocabulary):
    sum = 0
    for key, value in vocabulary.iteritems():
        sum += value[1]
    return sum

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
    index = 1
    for line in lines:
        arr = line.rstrip().split(' ')
        vocabulary[index] = arr[1]
        index += 1
    return vocabulary

def read400Terms():
    lines = readFile("/classic400/wordlist.txt")
    vocabulary = {}
    index = 1
    for line in lines:
        line = line.rstrip()
        vocabulary[index] = line
        index += 1
    return vocabulary

def read400TrueLabels(documents):
    lines = readFile("/classic400/truelabels.txt")
    index = 0
    for line in lines:
        labels = line.split('\t')
        for label in labels:
            documents[index].category = int(label)
            index += 1


def read400():
    lines = readFile("/classic400/classic400.txt")
    documents = []
    for line in lines:
        line = line.rstrip().split('\t')
        doc = Document("", 0, [])
        for i in range(len(line)):
            count = int(line[i])
            for j in range(count):
                doc.words.append(i+1)
        documents.append(doc)
    read400TrueLabels(documents)
    return documents


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
            for j in range(int(arr[2])):
                doc.words.append(int(arr[1]))

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
    return documents

def readReuters(vocabulary, documents):
    path = "./data/reuters21578-xml/reut2-0"
    for i in range(22):
        if i < 10:
            filename = path + "0" + str(i)
        else:
            filename = path + str(i)
        filename += ".xml"
        parseFile(vocabulary, documents, filename)
    return {v:k for k, v in vocabulary.items()}


def parseFile(vocabulary, documents, filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    raw_docs = []

    for reuter, title, dateline, body in zip(root.iter('REUTERS'), root.iter('TITLE'), root.iter('DATELINE'), root.iter('BODY')):
        if (reuter.attrib['TOPICS'] == "YES"):
            fullText = title.text + " " + dateline.text + " " + body.text
            raw_docs.append(buildVocabulary(fullText, vocabulary))
    buildIndexes(vocabulary)

    for words in raw_docs:
        words = parseString(words, vocabulary)
        documents.append(Document("dok", 0, words))

def readStopWords():
    lines = readFile("stopwords.txt")
    stopwords = Set([])
    for line in lines:
        words = line.split(' ')
        for word in words:
            stopwords.add(word)
    return stopwords

def buildVocabulary(fullText, vocabulary):
    lines = fullText.split('\n')
    words = []
    stopwords = readStopWords()
    for line in lines:
        for word in re.findall(r"[\w']+", line):
            word = word.strip().rstrip().lstrip()
            if len(word) > 3 and not word.isdigit() and not word in stopwords:
                word = word.lower()
                word = stem(word)
                words.append(word)
                if word in vocabulary:
                    vocabulary[word] += 1
                else:
                    vocabulary[word] = 1
    return words

def buildIndexes(vocabulary):
    index = 1
    keysToRemove = []
    for key, value in vocabulary.iteritems():
        if value > 2:
            vocabulary[key] = index
            index += 1
        else:
            keysToRemove.append(key)
    for key in keysToRemove:
        vocabulary.pop(key, None)



def parseString(rawWords, vocabulary):
    words = []
    for word in rawWords:
        if word in vocabulary:
            words.append(vocabulary[word])
    return words

def removePunctation(s):
    return s.translate(table, string.punctuation)




class LDA:
    def __init__(self, K, alpha, beta, docs, V):
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
            for term in doc.words:
                z = numpy.random.randint(0, K)
                z_n.append(z)
                self.n_m_z[m, z] += 1
                self.n_z_t[z, term-1] += 1
                self.n_z[z] += 1
            self.z_m_n.append(numpy.array(z_n))

    def getTettas(self):
        tettas = []
        for doc in range(len(self.docs)):
            z_n = self.z_m_n[doc]
            tetta = [0 for i in range(self.K)]
            for z in z_n:
                tetta[z] += 1
            for i in range(len(tetta)):
                tetta[i] = tetta[i] / float(len(z_n))
            tettas.append(tetta)
        return tettas

    def inference(self):
        """learning once iteration"""
        for m, doc in enumerate(self.docs):
            z_n = self.z_m_n[m]
            n_m_z = self.n_m_z[m]
            n = 0
            n_m_z_sum = n_m_z.sum()
            for term in doc.words:
                # discount for n-th word t with topic z
                z = z_n[n]
                n_m_z[z] -= 1
                self.n_z_t[z, term-1] -= 1
                self.n_z[z] -= 1

                # sampling topic new_z for t
                p_z = self.n_z_t[:, term-1] * n_m_z / (self.n_z * n_m_z_sum)
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
            for term in doc.words:
                log_per -= numpy.log(numpy.inner(phi[:,term-1], theta))
            N += doc.length
        return numpy.exp(log_per / N)

def tettasToMatlab(tettas):
    return [[tettas[j][i] for j in range(len(tettas))] for i in range(len(tettas[0])) ]

def writeTettas(tettas, docs):
    newTettas = tettasToMatlab(tettas)
    dimensions = ["X = ", "Y = ", "Z = "]
    f = open("output.txt", "w")
    for tag, plots in zip(dimensions, newTettas):
        f.write(tag + str(plots))
    colors = "["
    for i in range(len(docs)):
        if docs[i].category == 1:
            colors += "[1, 0, 0];"
        elif docs[i].category == 2:
            colors += "[0, 1, 0];"
        else:
            colors += "[0, 0, 1];"
    f.write("C = " + str(colors) + "]")
    f.close()

def phiDifference(phi_prev, phi_curr):
	diff = phi_prev - phi_curr
	diff = diff * diff
	return diff.sum()
		
def lda_learning(lda, iteration, voca):
    phi_prev = lda.worddist()
    converged = False
    for i in range(iteration):
        print "before inference"
        lda.inference()
        print "before"
        phi_curr = lda.worddist()
        print "Difference: " + str(phiDifference(phi_prev, phi_curr))
        if phiDifference(phi_prev, phi_curr) < 0.00006 and converged == False:
            output_word_topic_dist(lda, voca)
            writeTettas(lda.getTettas(), lda.docs)
            converged = True
        phi_prev = phi_curr
    output_word_topic_dist(lda, voca)

def output_word_topic_dist(lda, voca):
    zcount = numpy.zeros(lda.K, dtype=int)
    wordcount = [dict() for k in xrange(lda.K)]
    for xlist, zlist in zip(lda.docs, lda.z_m_n):
        for x, z in zip(xlist.words, zlist):
            zcount[z] += 1
            if x in wordcount[z]:
                wordcount[z][x] += 1
            else:
                wordcount[z][x] = 1

    phi = lda.worddist()
    for k in xrange(lda.K):
        print ("\n-- topic: " + str(k) + " words: " + str(zcount[k]))
        for w in numpy.argsort(-phi[k])[:20]:
            print (str(voca[w+1]) + ": " + str(phi[k,w]) + " (" +str(wordcount[k].get(w+1,0)) + ")")

def main():
    k = 3
    alpha = 0.1
    beta = 0.1
    epochs = 200
    useClassic400 = True
    vocabulary = {}
    documents = []
    print vocabulary
    print "Number of docs: " + str(len(documents))
    print 

    if (not useClassic400):
        vocabulary = readReuters(vocabulary, documents)


        # documents = readFileNames()
        # vocabulary = readTerms()
        # readTermMatrix(documents)
    else:
        vocabulary = read400Terms()
        documents = read400()

    #numpy.random.seed(options.seed)

    lda = LDA(k, alpha, beta, documents, len(vocabulary))
    print ("corpus=" + str(len(vocabulary)) + ", K=" + str(k) + ", a=" + str(alpha) + ", b=" + str(beta))

    #import cProfile
    #cProfile.runctx('lda_learning(lda, options.iteration, voca)', globals(), locals(), 'lda.profile')
    lda_learning(lda, epochs, vocabulary)

if __name__ == "__main__":
    main()










