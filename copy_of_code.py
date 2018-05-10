# -*- coding: utf-8 -*-
from __future__ import division
import csv
import nltk
import re,os,math
import glob
import random,spacy
from nltk.util import ngrams
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

nlp=spacy.load('en_core_web_sm')

path="/home/pannaga/bel_project/500N-KPCrowd-v1/CorpusAndCrowdsourcingAnnotations/train/"
key_files=glob.glob(path+'*.key')
shuffled_candidate_key_files=glob.glob(path+'*-Shuffled')
#count_list=[]
keywords=[]
training_candidate_words=[]
text=[]
nounphrases=[]
trigrams=[]
named_entities=[]
file_head=[]
for file1 in key_files:
	file_head.append(file1.split('.')[0])
	file2=file1.split('.')[0]+'-Shuffled'
	lines=[]
	keyword=[]
	with open(file1,'r') as f1:
		print f1
		for line in f1:
			if (line.strip()!=''):
				lines.append(line.strip())
				keyword.append(line.strip())
	count=len(lines)
	#count_list.append(count)
	c=0
	with open(file2,'r') as f2:
		for line in f2:
			l=line.split('\t')[0]
			if((l not in lines) and (l!='') and (c<count)):
				lines.append(l)
				c+=1
	training_candidate_words.append(lines)
	keywords.append(keyword)
	token = []
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
		ne=list(doc.ents)
		ne_s=[]
		for w in ne:
			ne_s.append(str(w))
		named_entities.append(ne_s)
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

def score(word, word_list):
	
    value = 0.0
    for i, j in enumerate(word_list):
        if j == word:
            value += float(1 - (i / float(len(word_list))))
    return value

def capitalize(phrase):
	values={}
	reg1=r'[A-Z][a-z]+[A-Z][a-z]*'
	reg2=r'[A-Z][a-z]+'
	reg3=r'^[A-Z]{2,}[a-z]+'
	reg4=r'[a-z]+[A-Z][a-z]*'
	wor=phrase.split()
	tot=0
	if len(wor)!=0:
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
	else:
		return 0

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
	location = text[j].find(phrase)
	if location ==-1:
		return None
	return location

def difference_between_occurences(phrase,j):
	#print phrase
	list1=[]
	index=0
	avg = 0.0
	prev = 0.0
	while text[j].find(phrase,index) != -1:
		pos=text[j].find(phrase,index)
		#print(phrase)
		#print(pos)
		#some=input()
		index = pos + len(phrase)
		# print index
		if(len(list1) != 0):
			list1.append(pos - prev)
			print pos+" "+prev
			if len(list1) == 1:
				avg = 1.0/float(list1[0])
			else:
				avg = (avg + 1.0/(pos - prev))/2.0
				print avg
		prev = pos
		# print avg
	return avg

def named_entity(phrase,j):
	num_words=phrase.split()
	for k in num_words:
		if k in named_entities[j]:
			return 1
		else:
			return 0


def keyword_or_not(phrase, j):
	if phrase in keywords[j]:
		return 1
	else:
		return 0



#features=[]
fea=open('features.csv','w')
file_names=open('feature_data.csv','w')
fea_writer = csv.writer(fea)
name_writer = csv.writer(file_names)
fea_writer.writerow(["term_frequency","tfidf","capitalize"])
name_writer.writerow(["word","keyword_or_not"]) 
for i,file in enumerate(file_head):
	print(file)
	# file_names.write(file+' '+str(counter+1)+' '+str(len(training_candidate_words[i])+counter+1 )+ '\n')
	for w in training_candidate_words[i]:
		fea_writer.writerow([text[i].count(w),tfidf(w,text,text[i]),capitalize(w)])
		# print keywords[i]
		name_writer.writerow([w, keyword_or_not(w,i)])



# c=0
# counter=0
	#f=[OrderedDict((w,(text[i].count(w),tfidf(w,text,text[i]),len(w),capitalize(w),named_entity(w,i),noun_phrases(w,i),trigrams_tag(w,i),first_occur(w,i),difference_between_occurences(w,i))) for w in training_candidate_words[i])]
	#features.append(f)
#print features



'''count_vectorizer=CountVectorizer()
freq_term_mat=count_vectorizer.fit_transform(text)
print (count_vectorizer.vocabulary_)
tfidf_vectorizer=TfidfVectorizer()
tfidf_mat=tfidf_vectorizer.fit_transform(training_candidate_words)
idf=tfidf_vectorizer.idf_
print tfidf_vectorizer.vocabulary_['soap']
print (dict(zip(tfidf_vectorizer.get_feature_names(), idf)))'''

