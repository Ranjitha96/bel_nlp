from __future__ import division
import nltk
from nltk import word_tokenize
from bs4 import BeautifulSoup
import urllib
from nltk.corpus import stopwords
fh=open('/home/pannaga/bel project/corpus/citations_class/06_1.xml','r');
raw=fh.read().decode('utf8')		#reads the content from the file and uses utf8 encoding to decode
#print raw
soup=BeautifulSoup(raw,"lxml")		#to remove html tags,beautifulsoup is used
#print soup					
html_free=soup.get_text(strip=True)		#to get only the text from the url removing all the tags
#print html_free
tokens=word_tokenize(html_free)			#tokens=sent_tokenize(html_free)	#to tokenise words or sentences
clean_tokens=tokens[:]				#to remove stop words
sr=stopwords.words('english')
for token in tokens:
	if token in stopwords.words('english'):
		clean_tokens.remove(token)
freq=nltk.FreqDist(tokens)			#to find the frequency of tokens
for key,val in freq.items(): 			#prints each token and it's frequency
	print (str(key) + ':' + str(val))
