# -*- coding: utf-8 -*-
from __future__ import division
import csv
import numpy as np
import nltk
import re,os,math
import glob
import random,spacy
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

nlp=spacy.load('en_core_web_sm')
stopword = stopwords.words('english')
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
VERBS = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']
PRONOUNS = ['PRP', 'PRP$', 'WP', 'WP$']
REQ = NOUNS + VERBS + PRONOUNS

posToWN = {}
for noun in NOUNS:
    posToWN[noun] = wn.NOUN
for verb in VERBS:
    posToWN[verb] = wn.VERB

def tf(word,word_list):
	return word_list.count(word)/len(word_list)

def score(word, word_list):
    value = 0.0
    for i, j in enumerate(word_list):
        if j == word:
            value += float(1 - (i / float(len(word_list))))
    return value

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
tokens = []
rootwords=[]
relatedWords=[]
ps = PorterStemmer()
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
	text_file=file1.split('.')[0]+'.txt'
	with open(text_file,'r') as f:
		t=f.read()
		t=re.sub(r'\n',' ',t)
		t=re.sub(r'[^a-zA-Z0-9\' ]*','',t)
		token=nltk.word_tokenize(t)
        tags = nltk.pos_tag(token)
        filtered_wordsToTags = {w: s for w, s in tags if w not in stopword and s in REQ}
        stemmedToWords = {ps.stem(w): w for w in filtered_wordsToTags.keys()}
        synonymToStemmedWord = {}
        relatedWordSet = {}
        freq_count = {word: tf(word, token) for word in token}
        freq_score = {word: score(word, token) for word in token}

        posString = ''
        for word,tag in tags:
            if tag in posToWN.keys():
                posString += posToWN[tag]
            else:
                posString += '.'
        j=[]
        i = posString.find('n')
        j.append(i)
        str1 =""
        ind = []
        nouns = [""]
        while i < len(posString) and i != -1:
            if j[-1] == i - 1:
                nouns[-1] = " ".join(ind)
                relatedWordSet[token[i]] = [token[j[-1]]]
            else:
                nouns[-1] = " ".join(ind)
                str1 = ""
                ind =[]
                j=[]
                nouns.append("")
            j.append(i)
            ind.append(token[i])
            i = posString.find('n',i + 1)
        nouns.append(" ".join(ind))
        nouns = [x for x in nouns if x!='']

        for word in stemmedToWords.keys():
            if filtered_wordsToTags[stemmedToWords[word]] not in PRONOUNS:
                synset = wn.synsets(word, pos=posToWN[filtered_wordsToTags[stemmedToWords[word]]])
                flag = False
                for entry in synset:
                    for synonym in entry.lemma_names():
                        if synonym not in synonymToStemmedWord:
                            synonymToStemmedWord[synonym] = word
                            if word not in relatedWordSet:
                                relatedWordSet[word] = []
                        else:
                            if synonymToStemmedWord[synonym] != word:
                                relatedWordSet[synonymToStemmedWord[synonym]].append(word)
                            flag = True
                            break
                    if flag:
                        break
        relatedWords.append(relatedWordSet)
        rootWordFrequency = {}
        rootWordWeight = {}

        for root, list1 in relatedWordSet.iteritems():
            rootWordFrequency[root] = freq_count[stemmedToWords.get(root,root)]
            rootWordWeight[root] = freq_score[stemmedToWords.get(root,root)]
            for word in list1:
                rootWordFrequency[root] += freq_count[stemmedToWords.get(word,word)]
                rootWordWeight[root] += freq_score[stemmedToWords.get(word,word)]
		rootwords.append(rootWordWeight)
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
		tokens.append(token)
		f.close()
	text.append(t)

'''computes the TF-IDF score. It's the product of tf and idf.'''
def tfidf(word,words_list1,word_list):
	return tf(word,word_list)*idf(word,words_list1)

'''computes "term frequency" which is the number of times a word appears in a document(here we have considered stemmed words of the document), normalized by dividing by the total number of words in document.'''

'''returns the number of documents containing word'''
def n_containing(word,words_lists):
	return sum(1 for doc in words_lists if word in doc)

'''computes "inverse document frequency" which measures how common a word is among all documents. The more common a word is, the lower its idf. We take the ratio of the total number of documents to the number of documents containing word, then take the log of that'''
def idf(word,words_lists):
	return math.log(len(words_lists)/(1+n_containing(word,words_lists)))

def score1(word, word_list):
	word = nltk.word_tokenize(word)
	value = []
	val=0
	for w in word:
		for i, j in enumerate(word_list):
			if j == w:
				val += float(1 - (i / float(len(word_list))))
		value.append(val)
	return sum(value)/len(value)
def capitalize(phrase):
	valu={}
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
					valu[wo]=1
				elif re.match(reg1,wo) or re.match(reg3,wo) or re.match(reg4,wo):
					valu[wo]=2
				elif  wo.isupper():
					valu[wo]=3
				else:
					if re.match(reg2,wo):
						valu[wo]=4
				tot+=valu[wo]
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

def root_word_score(phrase,j):
	num_words=phrase.split()
	for k in num_words:
		stem_k=ps.stem(k)
		for ph,list1 in relatedWords[j].iteritems():
			if stem_k in list1:
				key_val=ph
	return relatedWords[j][key_val]
'''def difference_between_occurences(phrase,j):
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
	return avg'''

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
fea_writer.writerow(["term_frequency","tfidf","capitalize","named entity","noun phrases","trigrams","Score"])
name_writer.writerow(["word","keyword_or_not"]) 
for i,file in enumerate(file_head):
	print(file)
	# file_names.write(file+' '+str(counter+1)+' '+str(len(training_candidate_words[i])+counter+1 )+ '\n')
	for w in training_candidate_words[i]:
		fea_writer.writerow([tf(w,tokens[i]),tfidf(w,tokens,tokens[i]),capitalize(w),named_entity(w,i),noun_phrases(w,i),trigrams_tag(w,i),score1(w,tokens[i]),root_word_score(w,i)])
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

