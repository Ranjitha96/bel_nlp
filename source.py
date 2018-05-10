import operator
import nltk,os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

stopword = stopwords.words('english')
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
VERBS = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']
# ADJS = ['JJ', 'JJR', 'JJS']
# ADVS = ['RB', 'RBR', 'RBS', 'WRB']
PRONOUNS = ['PRP', 'PRP$', 'WP', 'WP$']
REQ = NOUNS + VERBS + PRONOUNS

posToWN = {}
#print(wn.NOUN + ' ' + wn.VERB)
for noun in NOUNS:
    posToWN[noun] = wn.NOUN
for verb in VERBS:
    posToWN[verb] = wn.VERB


def tf(word, word_list):
    return word_list.count(word)


def score(word, word_list):
    value = 0.0
    for i, j in enumerate(word_list):
        if j == word:
            value += float(1 - (i / float(len(word_list))))
    return value

path='/home/pannaga/Desktop/random_text'
folder=os.listdir(path)
print folder
with open('lol','wb') as supercool:
    for i in folder:
        print i
        fil = open(path+"/"+i)
        supercool.write(bytes("%s "%(i)))
        text = fil.read().decode('utf-8')
        token = nltk.word_tokenize(text)
        ps = PorterStemmer()
        tags = nltk.pos_tag(token)
        filtered_wordsToTags = {w: s for w, s in tags if w not in stopword and s in REQ}
        # print(filtered_tags)
        stemmedToWords = {ps.stem(w): w for w in filtered_wordsToTags.keys()}
        # print(stemmed)
        #print(stemmedToWords)
        synonymToStemmedWord = {}
        relatedWordSet = {}
        # print token
        freq_count = {word: tf(word, token) for word in token}
        freq_score = {word: score(word, token) for word in token}
        #print (freq_count)
        #print(freq_score)

        posString = ''
        for word,tag in tags:
            if tag in posToWN.keys():
                posString += posToWN[tag]
            else:
                posString += '.'
        j=[]
        i = posString.find('n')
        j.append(i)
        str =""
        ind = []
        nouns = [""]
        while i < len(posString) and i != -1:
            #print(str(i) + ' ' + str(j))
            # comment : consider more than 2 nouns together... Chandler Muriel Bing +ve, -ve ??
            if j[-1] == i - 1:
                nouns[-1] = " ".join(ind)
                relatedWordSet[token[i]] = [token[j[-1]]]
            else:
                nouns[-1] = " ".join(ind)
                str = ""
                ind =[]
                j=[]
                nouns.append("")
            j.append(i)
            ind.append(token[i])
            i = posString.find('n',i + 1)
        nouns.append(" ".join(ind))
        nouns = [x for x in nouns if x!='']

        for word in stemmedToWords.keys():
            # print(word + " Stemmed")
            if filtered_wordsToTags[stemmedToWords[word]] not in PRONOUNS:
                # print(word + " Not Pronoun")
                synset = wn.synsets(word, pos=posToWN[filtered_wordsToTags[stemmedToWords[word]]])
                flag = False
                for entry in synset:
                    # print(str(entry) + " synset entry")
                    for synonym in entry.lemma_names():
                        # print(synonym + " synonym")
                        if synonym not in synonymToStemmedWord:
                            # print(word + " " + synonym + " New entry")
                            synonymToStemmedWord[synonym] = word
                            if word not in relatedWordSet:
                                relatedWordSet[word] = []
                        else:
                            #print(synonymToStemmedWord[synonym] + ' ' + word)
                            if synonymToStemmedWord[synonym] != word:
                                # print(word + " added to " + synonymToStemmedWord[synonym])
                                relatedWordSet[synonymToStemmedWord[synonym]].append(word)
                            flag = True
                            break
                    if flag:
                        break
        #print(relatedWordSet)

        rootWordFrequency = {}
        rootWordWeight = {}

        for root, list in relatedWordSet.iteritems():
            #if posToWN[filtered_wordsToTags[root]] != wn.NOUN:
            rootWordFrequency[root] = freq_count[stemmedToWords.get(root,root)]
            rootWordWeight[root] = freq_score[stemmedToWords.get(root,root)]
            #else:
                #rootWordFrequency[root] = freq_count[root]
            for word in list:
                rootWordFrequency[root] += freq_count[stemmedToWords.get(word,word)]
                rootWordWeight[root] += freq_score[stemmedToWords.get(word,word)]

        sortedroot = sorted(rootWordWeight.items(), key = operator.itemgetter(1),reverse=True)
        #print sortedroot
        count = 0
        for (word,scores) in sortedroot:
            if filtered_wordsToTags[stemmedToWords.get(word,word)] not in PRONOUNS:
                count+=1
                print ("%s - %f"%(word,scores))
                supercool.write(bytes("%s~%f "%(word,scores)))
            if count >10:
                break
        supercool.write(bytes("\n"))

#print(rootWordFrequency)
#print ("hello\n")
#print(rootWordWeight)

# zts = zip(tags,stemmed)
# print zts
# mapped_words = dict(zts)
# print(mapped_words)
# for stem in stemmed:
# 	if tags[stemmed.index(stem)]
# count = {}

# def mapper(wn_pos, nltk_pos):


# list_pos = NOUNS + VERBS + ADJS + ADVS
# print
# list_pos

#
# for stem in stemmed:
# 	wn.synsets(stem,pos= wn)
