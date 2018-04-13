# Import the necessary package to process data in JSON format
import tweepy
import pandas as pd

# Variables that contains the user credentials to access Twitter API
ACCESS_TOKEN = '89697138-GiBzcDsakKAqgC4sG530xcyFSO3fkQqdbIHDsQRYt'
ACCESS_SECRET = 'XdbuaNb7IlXOINaEi38wCQPkWDc9kt3j2gNEnA8xWsW7c'
CONSUMER_KEY = 'XRtBmWb1N4nQGOq31s2a5Ni8y'
CONSUMER_SECRET = 'r1s3kqOHQFSHhdha9Qcfo0Zf669DGiDqckGLfGLxdBIuRzUXwy'

# Initiate the connection to Twitter Search API
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.secure = True
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

api = tweepy.API(auth)
places = api.geo_search(query="USA", granularity="country")
place_id = places[0].id

# Search for latest tweets about "airlines"
tweets = []
for tweet in tweepy.Cursor(api.search, q='airlines', since='2015-01-01', result_type='recent', lang='en', count=10000, geo_enabled=True, geo_id='38.00, 97.00, 3000.37km').items(10000):
    tweets.append(tweet)

def toDataFrame(tweets):

    DataSet = pd.DataFrame()
    DataSet['tweetID'] = [tweet.id for tweet in tweets]
    DataSet['tweetText'] = [tweet.text for tweet in tweets]
    DataSet['tweetRetweetCt'] = [tweet.retweet_count for tweet in tweets]
    DataSet['tweetFavoriteCt'] = [tweet.favorite_count for tweet in tweets]
    DataSet['tweetSource'] = [tweet.source for tweet in tweets]
    DataSet['tweetCreated'] = [tweet.created_at for tweet in tweets]

    DataSet['userID'] = [tweet.user.id for tweet in tweets]
    DataSet['userScreen'] = [tweet.user.screen_name for tweet in tweets]
    DataSet['userName'] = [tweet.user.name for tweet in tweets]
    DataSet['userCreateDt'] = [tweet.user.created_at for tweet in tweets]
    DataSet['userDesc'] = [tweet.user.description for tweet in tweets]
    DataSet['userFollowerCt'] = [tweet.user.followers_count for tweet in tweets]
    DataSet['userFriendsCt'] = [tweet.user.friends_count for tweet in tweets]
    DataSet['userLocation'] = [tweet.user.location for tweet in tweets]
    DataSet['userTimezone'] = [tweet.user.time_zone for tweet in tweets]
    DataSet['geo'] = [tweet.geo for tweet in tweets]

    return DataSet

#Pass the tweets list to the above function to create a DataFrame
DataSet = toDataFrame(tweets)
DataSet.to_csv('airlineTweets.csv', sep=',', encoding='utf-8')