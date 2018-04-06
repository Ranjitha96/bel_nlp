# -*- coding: utf-8 -*-
from __future__ import division
import nltk
import re,os
import glob
import random
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

path="/home/pannaga/bel_project/500N-KPCrowd-v1/CorpusAndCrowdsourcingAnnotations/train/"
key_files=glob.glob(path+'*.key')
shuffled_candidate_key_files=glob.glob(path+'*-Shuffled')

training_candidate_words=[]
for file1 in key_files:
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
training_candidate_words=sum(training_candidate_words,[])
count_vectorizer=CountVectorizer()
freq_term_mat=count_vectorizer.fit_transform(training_candidate_words)
print len(count_vectorizer.vocabulary_)
tfidf_vectorizer=TfidfTransformer()
tfidf_mat=tfidf_vectorizer.fit_transform(freq_term_mat)
idf=tfidf_vectorizer.idf_
print len(dict(zip(count_vectorizer.get_feature_names(), idf)))
