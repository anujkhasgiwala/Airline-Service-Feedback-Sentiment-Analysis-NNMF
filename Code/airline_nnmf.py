from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

stops = stopwords.words('english')
data_negative = []
data_positive = []
def createDTM():
    with open('airline_tweets_sentiment_negative.txt', 'r', encoding="utf8") as negativeTweet:
        for row in negativeTweet.readlines():
            if len(row) > 1:
                data_negative.append(row)
        print(len(data_negative))
    with open('airline_tweets_sentiment_positive.txt', 'r', encoding="utf8") as positiveTweet:
        for row in positiveTweet.readlines():
            if len(row) > 1:
                data_positive.append(row)

def stopWordsRemoval():
    documents = []
    print('Reading negative tweets and cleaning')
    for line in data_negative:
        target = open('TopicNMF/negative.txt', 'w')
        documents.append(clean(line))
    if documents:
        nmfModel = buildingTopicModel(documents, target)

    documents = []
    print('Reading positive tweets and cleaning')
    for line in data_positive:
        target = open('TopicNMF/positive.txt', 'w')
        documents.append(clean(line))
    if documents:
        nmfModel = buildingTopicModel(documents, target)

def clean(line):
    text = line
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = word_tokenize(text)
    text = [w for w in text if w not in stops]
    text = [word for word in text if len(word) > 3]
    text = ' '.join([word for word in text])
    return text.strip()

def buildingTopicModel(documents, target):
    if len(documents) > 2:
        print('Creating Vectors')
        no_features = 1000
        # NMF is able to use tf-idf
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features)
        tfidf = tfidf_vectorizer.fit_transform(documents)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        print('Building Model')
        no_topics = len(documents)-1 if len(documents) < 10 else 10
        # Run NMF
        nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
        print('Display topics')
        # Display topics
        no_top_words = 10
        display_topics(nmf, tfidf_feature_names, no_top_words, target)
        target.close()
    return

def display_topics(model, feature_names, no_top_words, target):
    for topic_idx, topic in enumerate(model.components_):
        topic = " ".join([feature_names[i]
                        for i in topic.argsort()[: -no_top_words -1:-1]])
        target.write("Topic " + str(topic_idx)+":\n" + topic+"\n")

createDTM()
stopWordsRemoval()