# -*- coding: utf-8 -*-
from __future__ import division
import nltk
import re,os,math
import glob
import random,spacy
from nltk.util import ngrams
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

nlp=spacy.load('en_core_web_sm')

def tf(word,word_list,stem_words):
	return stem_words.count(word)/len(word_list)


path="/home/pannaga/bel_project/500N-KPCrowd-v1/CorpusAndCrowdsourcingAnnotations/train/"
key_files=glob.glob(path+'*.key')
shuffled_candidate_key_files=glob.glob(path+'*-Shuffled')

training_candidate_words=[]
text=[]
nounphrases=[]
trigrams=[]
file_head=[]
for file1 in key_files:
	file_head.append(file1.split('.')[0])
	file2=file1.split('.')[0]+'-Shuffled'
	lines=[]
	with open(file1,'r') as f1:
		for line in f1:
			lines.append(line.strip())
	count=len(lines)
	c=0
	with open(file2,'r') as f2:
		for line in f2:
			l=line.split('\t')[0]
			if((l not in lines) and (c<count)):
				lines.append(l)
				c+=1
	training_candidate_words.append(lines)
	text_file=file1.split('.')[0]+'.txt'
	with open(text_file,'r') as f:
		t=f.read()
		t=re.sub(r'\n',' ',t)
		t=re.sub(r'[^a-zA-Z0-9\' ]*','',t)
		token=nltk.word_tokenize(t)
		tri=ngrams(token,3)
		trigram=[]
		for i in tri:
			l=' '.join(i)
			trigram.append(l)
		trigrams.append(trigram)
		te=u"{}".format(t)
		doc=nlp(te)
		nounph=[]
		for np in doc.noun_chunks:
			nounph.append(np.text)
		nounphrases.append(' '.join(nounph))
		f.close()
	text.append(t)

'''computes the TF-IDF score. It's the product of tf and idf.'''
def tfidf(word,words_list1,word_list):
	return tf(word,word_list)*idf(word,words_list1)

'''computes "term frequency" which is the number of times a word appears in a document(here we have considered stemmed words of the document), normalized by dividing by the total number of words in document.'''
def tf(word,word_list):
	return word_list.count(word)/len(word_list)

'''returns the number of documents containing word'''
def n_containing(word,words_lists):
	return sum(1 for doc in words_lists if word in doc)

'''computes "inverse document frequency" which measures how common a word is among all documents. The more common a word is, the lower its idf. We take the ratio of the total number of documents to the number of documents containing word, then take the log of that'''
def idf(word,words_lists):
	return math.log(len(words_lists)/(1+n_containing(word,words_lists)))

def capitalize(phrase):
	values={}
	reg1=r'[A-Z][a-z]+[A-Z][a-z]*'
	reg2=r'[A-Z][a-z]+'
	reg3=r'^[A-Z]{2,}[a-z]+'
	reg4=r'[a-z]+[A-Z][a-z]*'
	wor=phrase.split()
	tot=0
	for wo in wor:
		if wo.isalpha():
			if wo.islower():
				values[wo]=1
			elif re.match(reg1,wo) or re.match(reg3,wo) or re.match(reg4,wo):
				values[wo]=2
			elif  wo.isupper():
				values[wo]=3
			else:
				if re.match(reg2,wo):
					values[wo]=4
			tot+=values[wo]
	return tot/len(wor)

def noun_phrases(phrase,j):
	if phrase in nounphrases[j]:
		return 1
	else:
		return 0

def trigrams_tag(phrase,j):
	if phrase in trigrams[j]:
		return 1
	else:
		return 0

def first_occur(phrase,j):
	return text[j].find(phrase)


features=[]
for i,file in enumerate(file_head[0:1]):
	print(file)
	#print training_candidate_words[i]
	f=[OrderedDict((w,(text[i].count(w),tfidf(w,text,text[i]),len(w),capitalize(w),noun_phrases(w,i),trigrams_tag(w,i),first_occur(w,i))) for w in training_candidate_words[i])]
	features.append(f)
#print features
averag = []
for items in features:
	for i in items:
		fisco={}
		for j in i.iteritems():
			val  = sum(j[1])
			fisco[j[0]]=val
		average =[]
		maxav  = max(fisco.values())
		average = {x[0]:x[1]/maxav for x in fisco.iteritems()}
		averag.append(average)

print averag

'''count_vectorizer=CountVectorizer()
freq_term_mat=count_vectorizer.fit_transform(text)
print (count_vectorizer.vocabulary_)
tfidf_vectorizer=TfidfVectorizer()
tfidf_mat=tfidf_vectorizer.fit_transform(training_candidate_words)
idf=tfidf_vectorizer.idf_
print tfidf_vectorizer.vocabulary_['soap']
print (dict(zip(tfidf_vectorizer.get_feature_names(), idf)))'''

