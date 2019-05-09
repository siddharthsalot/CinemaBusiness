# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:04:03 2019

@author: salots
"""

import glob
import os
import pandas as pd
from os import path
import sys
from textblob import TextBlob
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, words
from nltk.tokenize import WordPunctTokenizer
import configparser
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

DIRECTORY_PATH = "C:\\Users\\salots\\Documents\\Course Documents\\SNA\\Group_Project\\Test\\"
OUTPUT_PATH = DIRECTORY_PATH + "Sentiment_Output\\"
os.chdir(DIRECTORY_PATH)
TOKENIZER = WordPunctTokenizer()
PATTERN1 = r'@[A-Za-z0-9]+'
PATTERN2 = r'https?://[A-Za-z0-9./]+'
COMBINED_PATTERN = r'|'.join((PATTERN1, PATTERN2))
EXTENDED_LIST = ["twitter", "com","pic"]
stopWords = set(stopwords.words('english'))
config = configparser.ConfigParser()
config.read(os.getcwd() + "\\config.properties")
if(config.has_section("Common")):
    if(config.has_option("Common", "DIRECTORY_PATH")):
        DIRECTORY_PATH = config.get("Common", "DIRECTORY_PATH")
    if(config.has_option("Common","OUTPUT_PATH")):
        OUTPUT_PATH = config.get("Common", "OUTPUT_PATH")
    if(config.has_option("Common","EXTENDED_LIST")):
        EXTENDED_LIST = config.get("Common", "EXTENDED_LIST").split(",")

def SetCurrentDirectory(path):
    os.chdir(path)
    print("Current directory changed to: " + os.getcwd())
    return

def ExtractData(fileName):
    data = pd.read_csv(fileName, encoding="ISO-8859-1")
    data = data.loc[:,'url':'media']
    return data

def ExtractTweets(dataset):
    return dataset['content']

def CleanTweetText(tweet, extended_list, tokenize=False):
    soup = BeautifulSoup(tweet, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(COMBINED_PATTERN, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    if not tokenize:
        return clean
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    words = TOKENIZER.tokenize(lower_case)
    cleaned_tweet = (" ".join(words)).strip()
    words = word_tokenize(cleaned_tweet)
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            if w not in extended_list and len(w) > 1:
                wordsFiltered.append(w)

    return (" ".join(wordsFiltered))

def CleanTweets(tweets, extended_list):
    tweet_list = []
    for tweet in tweets:
        tweet_list.append(CleanTweetText(str(tweet),extended_list, True))
    return tweet_list

def GetOverallSentimentScore(SentimentList, threshold = 0):
    pos = 0
    neg = 0
    neu = 0
    pos1 = 0
    neg1 = 0 
    neu1 = 0
    pos5 = 0
    neg5 = 0 
    neu5 = 0
    for _,value in SentimentList.iteritems():
        if value > 0:
            pos += 1
        elif value < 0:
            neg += 1
        else:
            neu += 1
        
        if value > 0.01:
            pos1 += 1
        elif value < 0.01:
            neg1 += 1
        else:
            neu1 += 1
        
        if value > 0.05:
            pos5 += 1
        elif value < 0.05:
            neg5 += 1
        else:
            neu5 += 1
    
    return (round((pos/(pos+neg+neu)),2),round((pos1/(pos1+neg1+neu1)),2),round((pos5/(pos5+neg5+neu5)),2))

def GetAverageSentiment(dataset):
    dataset = dataset[dataset["Sentiment_Score"] != 0]
    return dataset["Sentiment_Score"].mean()

def CalcSentVaderScore(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    return sentiment_dict['compound']    

def Main():
    
    cleaned_tweets = []
    final_list = []
    threshold_1 = config.get("Common","Threshold_1", fallback=40)
    threshold_2 = config.get("Common","Threshold_2", fallback=80)
    for filepath in glob.iglob(DIRECTORY_PATH+"\\*.csv"):
        sentiment_dict = {}
        dataset = ExtractData(filepath)
        print(filepath)
        tweets = ExtractTweets(dataset)
        moviename = os.path.basename(filepath).split(".")[0]
        if(config.has_option(moviename,"EXTENDED_LIST")):
            extended_list = EXTENDED_LIST +\
                            config.get(moviename, "EXTENDED_LIST").split(",")
        else:
            extended_list = EXTENDED_LIST
        cleaned_tweets = CleanTweets(tweets, extended_list)
        dataset["Cleaned_Tweets"] = pd.DataFrame(tweet for tweet in cleaned_tweets)
        dataset.dropna(subset=['content'],inplace=True)
        dataset['Sentiment_Score'] = dataset['Cleaned_Tweets']\
                    .apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
        dataset['Subjectivity'] = dataset['Cleaned_Tweets']\
                .apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
        dataset['VaderScore'] = dataset['Cleaned_Tweets'].apply(CalcSentVaderScore)
        dataset[["content","Cleaned_Tweets","Sentiment_Score","Subjectivity","VaderScore"]]\
            .to_csv(OUTPUT_PATH + "\\" + os.path.basename(filepath), index=False)
        sentiment_dict["Movie"] = moviename
        sentiment_dict["Sentiment_0"],sentiment_dict["Sentiment_1"],sentiment_dict["Sentiment_5"] = GetOverallSentimentScore(dataset['Sentiment_Score'])
        sentiment_dict["SentimentAverage"] = GetAverageSentiment(dataset)
        if(config.has_option(moviename,"Star_Score")):
            sentiment_dict["Star_Score"] = config.get(moviename,"Star_Score", fallback=0)
        if(config.has_option(moviename,"Theatres")):
            sentiment_dict["Theatres"] = config.get(moviename,"Theatres", fallback=0)
        if(config.has_option(moviename,"Opening_Collection")):
            sentiment_dict["Opening_Collection"] = config.get(moviename,"Opening_Collection", fallback=0)
            if sentiment_dict["Opening_Collection"] < threshold_1:
                sentiment_dict["Collection_Label"] = 0
            elif threshold_1 < sentiment_dict["Opening_Collection"] < threshold_2:
                sentiment_dict["Collection_Label"] = 1
            elif sentiment_dict["Opening_Collection"] > threshold_2:
                sentiment_dict["Collection_Label"] = 2
        if(config.has_option(moviename,"Rank")):
            sentiment_dict["Rank"] = config.get(moviename,"Rank", fallback=0)
        if(config.has_option(moviename,"Rating")):
            sentiment_dict["Rating"] = config.get(moviename,"Rating", fallback=0)
        final_list.append(sentiment_dict)

    sentiment_data = pd.DataFrame(final_list)
    return sentiment_data

if __name__ == '__main__':
    sentiment_data = Main()
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    sentiment_data.to_csv(OUTPUT_PATH + "\\Output.csv", index=False)