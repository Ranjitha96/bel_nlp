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
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
stop=stopwords.words('english')
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']

def get_continuous_chunks(text):
	chunked = ne_chunk(pos_tag(word_tokenize(text)))
	prev = None
	continuous_chunk = []
	current_chunk = []
	for i in chunked:
		if type(i) == Tree:
			current_chunk.append(" ".join([token for token, pos in i.leaves()]))
		elif current_chunk:
			named_entity = " ".join(current_chunk)
			if named_entity not in continuous_chunk:
				continuous_chunk.append(named_entity)
				current_chunk = []
		else:
			continue
	return continuous_chunk

text="Canadian pop star Michael Buble kisses his bride Argentine TV actress Luisana Lopilato after their civil wedding ceremony in Buenos Aires, Argentina, Thursday March 31, 2011. (AP Photo/Natacha Pisarenko) BUENOS AIRES, Argentina (AP) - Canadian pop star Michael Buble married Argentine TV actress Luisana Lopilato in a civil ceremony on Thursday. The Grammy-winning singer of ""Crazy Love"" and his Argentine sweetheart posed for a mob of fans after tying the knot at a civil registry in downtown Buenos Aires. She wore a lilac chiffon dress with silver high heels and Buble wore a sharp gray suit as he leaned down for a smooch. Then Lopilato, 23, tossed a bouquet of purple orchids into the crowd as some fans threw rice and red rose petals and a young woman shouted ""I love you Michael!"" Buble, 35, won his Grammy last month for traditional pop vocal album, and is one of North America's top-grossing concert entertainers. Lopilato made her name as a model and in Argentine sitcoms and soap operas, including ""Rebel Way,"" ""Little Girls,"" ""Married With Children"" and ""Pirate Soul."" The couple plan a full ceremony with 300 guests next month at a mansion outside Buenos Aires, and another wedding in Vancouver in April. They have homes in Canada, Los Angeles and Buenos Aires province. Buble dated British actress Emily Blunt for several years before he met Lopilato in 2009 during a South American concert tour. Lopilato was romantically involved before with tennis player Juan Monaco. With more than 500,000 fans and followers in Facebook and Twitter, Lopilato sent messages throughout the day, thanking her family and the media. ""How beautiful it all was!"" she wrote. "
print get_continuous_chunks(text)
