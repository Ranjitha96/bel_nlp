import nltk,re
text1="""Canadian pop star Michael Buble kisses his bride Argentine TV actress Luisana Lopilato after their civil wedding ceremony in Buenos Aires, Argentina, Thursday March 31, 2011. (AP Photo/Natacha Pisarenko) BUENOS AIRES, Argentina (AP) - Canadian pop star Michael Buble married Argentine TV actress Luisana Lopilato in a civil ceremony on Thursday. The Grammy-winning singer of ""Crazy Love"" and his Argentine sweetheart posed for a mob of fans after tying the knot at a civil registry in downtown Buenos Aires. She wore a lilac chiffon dress with silver high heels and Buble wore a sharp gray suit as he leaned down for a smooch. Then Lopilato, 23, tossed a bouquet of purple orchids into the crowd as some fans threw rice and red rose petals and a young woman shouted ""I love you Michael!"" Buble, 35, won his Grammy last month for traditional pop vocal album, and is one of North America's top-grossing concert entertainers. Lopilato made her name as a model and in Argentine sitcoms and soap operas, including ""Rebel Way,"" ""Little Girls,"" ""Married With Children"" and ""Pirate Soul."" The couple plan a full ceremony with 300 guests next month at a mansion outside Buenos Aires, and another wedding in Vancouver in April. They have homes in Canada, Los Angeles and Buenos Aires province. Buble dated British actress Emily Blunt for several years before he met Lopilato in 2009 during a South American concert tour. Lopilato was romantically involved before with tennis player Juan Monaco. With more than 500,000 fans and followers in Facebook and Twitter, Lopilato sent messages throughout the day, thanking her family and the media. ""How beautiful it all was!"" she wrote."""
'''def difference_between_occurences(phrase):
	list1=[]
	index=0
	avg = 0.0
	prev = 0.0
	while text1.find(phrase,index) != -1:
		pos=text1.find(phrase,index)
		index = pos + len(phrase)
		print len(list1)
		if(len(list1) != 0):
			list1.append(pos - prev)
			print pos
			if len(list1) == 1:
				avg = 1.0/float(list1[0])
			else:
				avg = (avg + 1.0/(pos - prev))/2.0
				print avg
		prev = pos
		# print avg
	return avg

difference_between_occurences('Michael')'''
text1=re.sub(r'[^a-zA-Z0-9\' ]*','',text1)
def score(word, word_list):
	word = nltk.word_tokenize(word)
	value = []
	val=0
	for w in word:
		for i, j in enumerate(word_list):
			if j == w:
				val += float(1 - (i / float(len(word_list))))
		value.append(val)
	return sum(value)/len(value)

score('Michael',nltk.word_tokenize(text1))
