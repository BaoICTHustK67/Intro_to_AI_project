from nltk.corpus import stopwords
import nltk
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
stopWord = set(stopwords.words('english'))

def get_importantFeature(sent):
    sent = sent.lower()

    returnList = []
    sent = nltk.word_tokenize(sent)
    for i in sent:
        if i.isalnum():
            returnList.append(i)
    return returnList 


def removing_stopWord(sent):
    returnList = []
    for i in sent:
        if i not in stopWord and i not in string.punctuation:
            returnList.append(i)
    return returnList


def potter_stem(sent):
    returnList = []
    for i in sent:
        returnList.append(ps.stem(i))
    return " ".join(returnList)

