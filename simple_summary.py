from __future__ import division
import nltk
import re,os,math
from nltk import word_tokenize,pos_tag,sent_tokenize
from collections import OrderedDict
from nltk.corpus import stopwords
from textblob import TextBlob as tb
from nltk.stem.porter import PorterStemmer
stop_words=stopwords.words('english')
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
VERBS = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']

'''computes the TF-IDF score. It's the product of tf and idf.'''
def tfidf(word,words_list1,word_list,stem_words):
	return tf(word,word_list,stem_words)*idf(word,words_list1)

'''computes "term frequency" which is the number of times a word appears in a document(here we have considered stemmed words of the document), normalized by dividing by the total number of words in document excluding stopwords.'''
def tf(word,word_list,stem_words):
	return stem_words.count(word)/len(word_list)

'''returns the number of documents containing word'''
def n_containing(word,words_lists):
	return sum(1 for doc in words_lists if word in doc)

'''computes "inverse document frequency" which measures how common a word is among all documents. The more common a word is, the lower its idf. We take the ratio of the total number of documents to the number of documents containing word, then take the log of that'''
def idf(word,words_lists):
	return math.log(len(words_lists)/(n_containing(word,words_lists)))


stemmer=PorterStemmer()	#initialising the stemmer


doc_list=[]

'''getting the contents from the document and initalising it with Textblob and storing it in list'''
documents=['/home/pannaga/bel project/corpus/dataset2/insidious1.txt','/home/pannaga/bel project/corpus/dataset2/insidious2.txt','/home/pannaga/bel project/corpus/dataset2/insidious3.txt','/home/pannaga/bel project/corpus/dataset2/insidious4.txt']
for doc in documents:
	file=open(doc,'r')
	cont=file.read().encode('utf-8').lower().replace("'",'')
	doc_list.append(tb(cont))


sent_list=[]
words_list=[]
tags_list=[]

'''Tokenization and removal of stop words'''
for text in doc_list:
	sent_token=text.sentences
	sent_list.append(sent_token)
	word_tokens=text.words
	tags=text.tags
	filtered_words=[w for w in word_tokens if not w in stop_words]
	filtered_tags=[t for t in tags if not t[0] in stop_words]
	tags_list.append(filtered_tags)
	words_list.append(filtered_words)


nouns_and_verbs_list=[]

'''extracting only nouns and verbs'''
for tags in tags_list:
	noun_or_verb_words=[item[0] for item in tags if ((item[1] in NOUNS) or (item[1] in VERBS))]
	nouns_and_verbs_list.append(noun_or_verb_words)
	

stemmed_words=[]
stemmed_nv=[]

'''stemming of words '''
for tokens in words_list:
	stemmed_token=[stemmer.stem(token) for token in tokens]
	stemmed_words.append(stemmed_token)

for nvs in nouns_and_verbs_list:
	nv_token=[stemmer.stem(nv) for nv in nvs]
	stemmed_nv.append(nv_token)


score_doc=[]

'''calculating the tfidf values for all nouns and verbs'''
for i,text in enumerate(doc_list):
	score_words={word:tfidf(word,stemmed_words,stemmed_words[i],stemmed_nv[i]) for word in stemmed_words[i]}
	score_doc.append(score_words)


sorted_scores=[]

'''soting the scores in descending order '''
for scores in score_doc:
	sorted_score=OrderedDict(sorted(scores.items(),key=lambda t:t[1],reverse=True))
	sorted_scores.append(sorted_score)


scored_sentences=[]

'''Calculating the sentencce importance'''
for i,sentences in enumerate(sent_list):
	sent_score={}
	for sentence in sentences:
		score=0
		sentence_tag=sentence.tags
		filtered_sentence_tag=[t for t in sentence_tag if not t[0] in stop_words]
		for each_tag in filtered_sentence_tag:
			if (each_tag[1] in NOUNS) or (each_tag[1] in VERBS):
				sw=stemmer.stem(each_tag[0])
				score=score+score_doc[i][sw]
		sent_score.update({sentence:score})
	scored_sentences.append(sent_score)


sorted_doc=[]

'''sorting the sentence in descending order'''
for sentences in scored_sentences:
	sorted_sent=OrderedDict(sorted(sentences.items(),key=lambda t:t[1],reverse=True))
	sorted_doc.append(sorted_sent)


'''selecting a first few sentences(depending on compression) from the sorted list of sentences '''
compression_rate=5

comp_stats_list=[]
for doc in sorted_doc:
	compressed=doc.items()[0:5]
	comp_stats=[s[0] for s in compressed]
	comp_stats_list.append(comp_stats)


summary=[]

'''summarizing it in proper order'''
for i,sen in enumerate(sent_list):
	string=''
	for st in sen:
		if st in comp_stats_list[i]:
			string+=str(st)
	summary.append(string)
print summary[0]

	

