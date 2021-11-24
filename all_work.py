# libs
import streamlit as st
import pandas as pd
import numpy as np

import tweepy
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

import nltk
from nltk import text
from nltk.tree import Tree
from nltk.corpus import stopwords
from collections import Counter

from tweepy import streaming
import  json
import time
from tweepy import auth

import streamlit as st
import pandas as pd
import numpy as np



stop_words = stopwords.words('english')
stemmer = nltk.SnowballStemmer("english")

# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'
         }

#---------------------------------------------------------------------------#
#-------------------------------Variables--------------------------------#
tweet, text, cleaned_text, sentiment = str , str, str, str
streamlit_text, streamlit_sentiment = str, str
selected_key_words , search_key_words= [], []
num_of_tweets = 10
list_of_cleaned_text, list_of_stream_tweets , list_of_stream_sentiment = [], [], []
done = False


#---------------------------------------------------------------------------#
#-------------------------------Twitter keys--------------------------------#

# Mostafa Nabieh Key's
TWITTER_CONSUMER_KEY = "hdbxpzj6UNQJ8AlsncMXfhZPy"
TWITTER_CONSUMER_SECRET = "ixf6yFYlhSRgMllednsuzcZQdbFEo5ROUBqKAHGQj3x5ugeqVE"
TWITTER_ACCESS_TOKEN = "1437601077998784512-f2xEgL4IQKyHqUsX0TdV6JndZ4kWsa"
TWITTER_ACCESS_TOKEN_SECRET = "g0PlCwd4CEaRlTxK5NVCYyznh3qfGB22HLsCZOEvXjZ3P"



#----------------------------------------------------------------------------------#
#-------------------------------Starting Streamlit--------------------------------#
st.title('Hello world from Twitter streaming app!')
# st.write('Wait for getting the smal batch of streamming tweets...')


# search_key_words = st.text_input('Enter your keywords for extracting tweets:')

# st.write(f'The key words seache: {search_key_words}')

selected_key_words = st.sidebar.multiselect(
    'Select keywords for make sentiment on it, and when you are finished, click Done!',
    ['Movies', 'Series', 'Sports','Corona', 'COVID-19', 'Egypt', 'Football', 'Mo Salah'])


if st.sidebar.button('Selecting Done!'):
    done = True
    # st.write(f'You select: {selected_key_words}')
    search_key_words = selected_key_words 
    st.write('## Tweet And Sentiment for Selected keywords:')

entering_key_words = st.text_input('Enter your keywords for extracting tweets, and when you are finished, click Entering Done!')
if st.button('Entering Done!'):
    done = True
    # st.write(f'You select: {entering_key_words}')
    search_key_words = entering_key_words.split()
    st.write('## Tweet And Sentiment for Selected keywords:')
    
    

st.write(f'The key words search: {search_key_words}')
# st.write(f'The key words search: {entering_key_words}')
# search_key_words = [entering_key_words.split()]
# print(type(search_key_words))
# print(search_key_words)
# st.write('The default search keywords are: "movies", "series", "Robert_De_Niro", "The_Comeback_Trail"')


#----------------------------------------------------------------------------------#
#-------------------------------Important functoins--------------------------------#
def clean_text(text):
    '''
        Make text lowercase, remove text in square brackets,remove links,remove punctuation
        and remove words containing numbers.
    '''
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    text = re.sub(r"[^A-Za-z\s]", "", text.strip())                 # take just english tweets
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)                # remove urls
    text = re.sub(r'\s+', ' ', text)                                # Removing multiple spaces
    text = re.sub(r'@[^\s]+', ' ', text)                            # Removing @user
    text = re.sub(r'#', ' ', text)                                  # Removing #
    text = re.sub(r'rt', ' ', text)                                 # Removing RT
    text = re.sub(sequencePattern, seqReplacePattern, text)         # Replace 3 or more consecutive letters by 2 letter.
    text = re.sub('<.*?>+', '', text)                               # remove tages
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # remove punctuation
    
    text = re.sub('\n', '', text)                                   # remove new line
    text = re.sub('\w*\d\w*', '', text)
    return text

#---------------------------------------------------------------------------#

def preprocess_data(text):    
    # Clean puntuation, urls, and so on
    text = clean_text(text)
    
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)    
    # Replace all emojis.
    if ' '.join(emojis[word] + 'EMOJI' for word in text.split() if word in emojis.keys()):
        text = ' '.join(emojis[word] + 'EMOJI' for word in text.split() if word in emojis.keys()) 
    # Stem all the words in the sentence
    text = ' '.join(stemmer.stem(word) for word in text.split()) 
    return text
#---------------------------------------------------------------------------#

def one_predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform([preprocess_data(text)])
    # textdata = vectoriser.fit_transform(clean_text(text))
    sentiment = model.predict(textdata)[0]
    return "Negative" if sentiment == 0 else "Positive"

def multiple_predict(vectoriser, model, texts):
    # Make a list of text with sentiment.
    data = []
    for text in texts:
        # Predict the sentiment
        textdata = vectoriser.transform([clean_text(text)])
        sentiment = model.predict(textdata)[0]
        data.append((text,sentiment))
        
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df

#---------------------------------------------------------------------------#

# Load the model and tf-idf 
def load_models():
    '''
    Replace '..path/' by the path of the saved models.
    '''
    
    # Load the vectoriser.
    
    file = open('./vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()

    # Load the LSVC Model.
    file = open('./Sentiment-LSVC.pickle', 'rb')
    LSVC_model = pickle.load(file)
    file.close()
    
    return vectoriser, LSVC_model



# def take_tweet(text):
#     sentiment = one_predict(vectoriser, LinearSVC, text)
#     return text, sentiment

# Loading the models.
vectoriser, LinearSVC = load_models()
print("We will loading the LinearSVC model and tf-idf embedding!")

#---------------------------------------------------------------------------#
#-------------------------------Streaming functions--------------------------#

# 1- Creat a SteamListener
class MaXListener(tweepy.StreamListener):
    def on_data(self, raw_data):
        # -------- Keys() --------
        # dict_keys(['created_at', 'id', 'id_str', 'text', 'source', 'truncated', 'in_reply_to_status_id', 'in_reply_to_status_id_str', 'in_reply_to_user_id', 'in_reply_to_user_id_str', 'in_reply_to_screen_name', 'user', 'geo', 'coordinates', 'place', 'contributors', 'retweeted_status', 'is_quote_status', 'quote_count', 'reply_count', 'retweet_count', 'favorite_count', 'entities', 'favorited', 'retweeted', 'filter_level', 'lang', 'timestamp_ms'])
        cnt = 1
        try:
            msg = json.loads(raw_data)
            text = msg['text']
            
            print('-'*50, cnt, "-new message", '-'*50,)
            print(f"The Original text:\n{text}")

            cleaned_text = preprocess_data(text)
            # in this func, it's already make the text preprocessing for modeling.
            sentiment = one_predict(vectoriser, LinearSVC, text)
            
            # adding each tweet and sentiment in the dataframe
            if cleaned_text and sentiment:    
                list_of_stream_tweets.append(text)
                list_of_stream_sentiment.append(sentiment)
                list_of_cleaned_text.append(cleaned_text)

                # show the df in stramlit
                df = pd.DataFrame({'Tweet': list_of_stream_tweets,
                                    # 'Cleaned_Tweet': list_of_cleaned_text,
                                    'Sentiment': list_of_stream_sentiment})
            st.table(df)
            cnt += 1 
            print(f'The clean text:\n{cleaned_text}\nThe prediction: {sentiment}')

            
            time.sleep(5)
            # return True if cnt < num_of_tweets else False
            # return True
            if cnt > num_of_tweets:
                return False
            else: 
                True
        except BaseException as e:
            print(f"Error on_data: {str(e)}")
        return True

    def process_data(self, raw_data):
        print(raw_data)
        
    def on_error(self, status_code):
        if status_code == 420:
            # return false in on_data to disconnect the streaming!
            return False

#---------------------------------------------------------------------------#
# 2- Creat a Stream
class MaxStream():
    def __init__(self, auth, listener):
        self.stream = tweepy.Stream(auth= auth, listener= listener)

    def start(self, keywords_list):
        self.stream.filter(track=keywords_list)
#---------------------------------------------------------------------------#

# 3- Start Stream
if __name__ == "__main__":
    listener = MaXListener ()
    auth = tweepy.OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
    auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
    
    if done and search_key_words:
        # print(search_key_words)
        stream = MaxStream(auth, listener)
        stream.start(search_key_words)

#---------------------------------------------------------------------------#