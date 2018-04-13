from textblob import TextBlob
from codecs import *
from nltk import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import codecs
import csv

def sentiment(message):
    blob = TextBlob(message)
    if blob.sentiment.polarity > 0:
        return 'Positive'
    elif blob.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

def removeLink(message):
    x=re.sub("(https://[^ ]+)", "", message)
    return x

def removeAt(message):
    x = re.sub(r'(\s)@\w+', r'\1', message)
    return x

def removeHash(message):
    x = re.sub(r'#(\w+)', r'\1', message)
    return x

#csv file reading
def csvReader():
    with open('shobhitreviews.csv','rb', encoding="utf8") as tweetCSV:
        csvFile = csv.reader(tweetCSV)
        for row in csvFile:
            t = removeHash(removeAt(removeLink(row[0])))
            x = sentiment(t)
            with open('shobhitreviews_sentiment.csv', 'a', encoding="utf8") as tweetCSV1:
                csvFileWrite = csv.writer(tweetCSV1,delimiter = ',')
                csvFileWrite.writerow((row[0],row[1],row[2],row[3],x))

csvReader()