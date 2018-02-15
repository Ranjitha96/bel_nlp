# -*- coding: utf-8 -*-
from __future__ import division
import nltk
import re,os
from nltk import word_tokenize,pos_tag
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import urllib
from nltk.corpus import stopwords
import pandas as pd
stop=stopwords.words('english')
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']

class document:
	path=""
	content=""
	def __init__(self,pname):
		self.path=pname

	def read_content(self):
		fh=open(self.path,'r')
		self.content=fh.read()
		return self.content

	



if __name__ == '__main__':
	path_liist=['/home/pannaga/bel project/corpus/dataset2/insidious1.txt','/home/pannaga/bel project/corpus/dataset2/insidious2.txt','/home/pannaga/bel project/corpus/dataset2/insidious3.txt','/home/pannaga/bel project/corpus/dataset2/insidious4.txt']
	docs=[document(pathname) for i in range(0,4) for pathname in path_liist]
	contents=[doc.read_content() for doc in docs]
	count_vect=CountVectorizer(stop_words='english')					#Convert a collection of text documents to a matrix of token counts
	train_vect=count_vect.fit_transform(contents)						#Learn a vocabulary dictionary of all tokens in the raw documents.
	train_vect_df=pd.DataFrame(train_vect.toarray(),columns=count_vect.get_feature_names())	#converting the document-term matrix into a dataframe
	print train_vect_df
	"""tfidf=TfidfTransformer(norm="l2")	#Transform a count matrix to a normalized tf or tf-idf representation
	train_tfidf=tfidf.fit_transform(train_vect)"""
	



