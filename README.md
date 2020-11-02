# Tweet-Sentiment-Extraction

# Introduction

Sentiment Analysis can be defined as the process of analyzing text data and categorizing them into Positive, Negative, or Neutral sentiments. Sentiment Analysis is used in many cases like Social Media Monitoring, Customer service, Brand Monitoring, political campaigns, etc. Analyzing customer feedback such as social media conversations, product reviews, and survey responses allows companies to understand the customer's emotions better which is becoming more essential to meet their needs.


# Usage of ML/DL for this problem


It is almost impossible to manually sort thousands of social media conversations, customer reviews, and surveys. So we have to use either ML/DL to build a model that analyzes the text data and performs the required operations. The problem I am trying to solve here is part of this Kaggle competition. In this problem, we are given some text data along with their sentiment(positive/negative/neutral) and we need to find the phrases/words that best supports the sentiment.


# Data Overview

The dataset used here is from the Kaggle competition Tweet Sentiment Extraction. The dataset used in this competition is from phrases from Figure Eight's Data for Everyone platform. It consists of two data files train.csv and test.csv, where there are 27481 rows in training data and 3534 rows in test data.

# List of columns in the dataset

textID: unique id for each row of data

text: this column contains text data of the tweet.

sentiment: the sentiment of the text (positive/negative/neutral)

selected_text: phrases /words from the text that best supports the sentiment

