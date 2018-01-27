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
stop=stopwords.words('english')
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']

def preprocessing(document):
	soup=BeautifulSoup(document,"lxml")		#to remove html tags,beautifulsoup is used
	html_free=soup.get_text(strip=True)		#to get only the text from the url removing all the tags
	processing=re.sub('[^A-Za-z .-]+',' ', html_free)							#removing all punctuation marks ,non alpha numeric values 
	processing=processing.replace('-',' ').replace('.',' ').replace('...',' ')  #removing full stops and ellipses
	return processing

def make_list():											#cextracting the data from each training file in the folder and storing it in a list.
	srcdir='/home/pannaga/bel project/corpus/train_data'	#we are storing the training data as list because CountVectoriser.fit() requires it's parameter to be a list
	allfiles=os.listdir(srcdir)
	training_list=[]
	for i in allfiles:
		file=open(os.path.join(srcdir,i),'r')
		cont=file.read()
		training_list.append(cont)
	return training_list

def tokenize(document):
	words=nltk.tokenize.word_tokenize(document)					#tokenising words
	words=[word.lower() for word in words if word not in stop] 	#removing stop words
	return words
	
#def extract_subject(document):
#	freqdist=word_freq_dist(document)
#	most_freq_nouns=[w for w,c in fdist.most_common(10) if nltk.pos_tag([w])[0][1] in NOUNS]

if __name__ == '__main__':
	fh=open('/home/pannaga/bel project/corpus/citations_class/06_1.xml','r')
	raw=fh.read().decode('utf8')		#reads the content from the file and uses utf8 encoding to decode
	processed=preprocessing(raw)
	tokens=tokenize(processed)
	tags=pos_tag(words)
	fdist=nltk.FreqDist(words)			#finding frequency distribution of words
	training_data=make_list()			#coverting training files into list
	count_vect=CountVectorizer()		#Convert a collection of text documents to a matrix of token counts
	count_vect=count_vect.fit(training_data)	#Learn a vocabulary dictionary of all tokens in the raw documents.
	freq_term_matrix=count_vect.transform(training_data)	#Transform documents to document-term matrix.
	tfidf=TfidfTransformer(norm="l2")	#Transform a count matrix to a normalized tf or tf-idf representation
	tfidf.fit(freq_term_matrix)			#Learn the idf vector (global term weights)
	doc_freq_term=count_vect.transform([processed])
	doc_tfidf_matrix=tfidf.transform(doc_freq_term)	#Transform a count matrix to a tf or tf-idf representation