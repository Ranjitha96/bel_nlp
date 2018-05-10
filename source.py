import operator
import nltk,os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

stopword = stopwords.words('english')

l

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
        str =""
        ind = []
        nouns = [""]
        while i < len(posString) and i != -1:
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
        rootWordFrequency = {}
        rootWordWeight = {}

        for root, list in relatedWordSet.iteritems():
            rootWordFrequency[root] = freq_count[stemmedToWords.get(root,root)]
            rootWordWeight[root] = freq_score[stemmedToWords.get(root,root)]
            for word in list:
                rootWordFrequency[root] += freq_count[stemmedToWords.get(word,word)]
                rootWordWeight[root] += freq_score[stemmedToWords.get(word,word)]

        sortedroot = sorted(rootWordWeight.items(), key = operator.itemgetter(1),reverse=True)
        count = 0
        for (word,scores) in sortedroot:
            if filtered_wordsToTags[stemmedToWords.get(word,word)] not in PRONOUNS:
                count+=1
                print ("%s - %f"%(word,scores))
                supercool.write(bytes("%s~%f "%(word,scores)))
            if count >10:
                break
        supercool.write(bytes("\n"))

