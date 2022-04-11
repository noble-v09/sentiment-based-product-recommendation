import pandas as pd
import numpy as np
import pickle
# from concurrent.futures import ThreadPoolExecutor
# import threading
# import asyncio
# import swifter
# from preprocessor import *

## Import model
with open('./models/model.pkl', 'rb') as f:
    model = pickle.load(f)

## Import recommendations matrix
with open('./models/user_recommendations.pkl', 'rb') as f:
    user_ratings = pickle.load(f)

## Import tfidf vectorizer
with open('./models/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

## Get user_list from recommendations model
def get_user_list():
    user_list = list(user_ratings.index.unique())
    ## Prepending --SELECT-- as first option
    user_list.insert(0, "--SELECT--")
    return user_list


## Text Preprocessing
## Ideally, text preprocessing is ideally done in following ways:
## Case 1: We do the preprocessing step directly in DB/File itself and keep it ready, rather than doing it at runtime
## Case 2: App Startup
## Case 3: At runtime, as soon as api is called and the data is read from DB/file
## Read reviews
# reviews = pd.read_csv('./data/sample30.csv', usecols=['name', 'reviews_text'])
# def preprocess():
# reviews['reviews_text'] = preprocess_text(reviews['reviews_text'])
# reviews['reviews_text'] = reviews['reviews_text'].apply(lemmatize_text)
# t1 = threading.Thread(target=preprocess)
# t1.start()

## Due to resource constraints in Heroku Deployment(constantly getting "R14: Memory quota exceeded" error), 
## here the reviews dataset is already preprocessed 
## and clean dataset ==> ['reviews_text', 'product_name'] is being used
with open('./models/clean_df.pkl', 'rb') as f:
    reviews = pickle.load(f)

## Function to generate sentiment for each product recommended
def generate_sentiment(product):
    '''Function to generate sentiment for each product recommended by model'''
    prod_reviews = reviews.loc[reviews['name']==product, ['reviews_text']]
            
    ## Apply TFIDF Vectorizer to reviews
    X = pd.DataFrame(data=tfidf.transform(prod_reviews['reviews_text']).toarray(), columns=tfidf.get_feature_names())

    ## Predict Sentiment --> Negative == 0, Positive == 1
    y = model.predict(X)
    return (product, round(np.sum(y)/len(y)*100, 2))

def get_top_5_recommendations(user):
    ## Get top 20 recommendations    
    recommendations = user_ratings.loc[user].sort_values(ascending=False)[:20]

    # with ThreadPoolExecutor() as executor:
    #     results = executor.map(generate_sentiment, recommendations.index)
    results = map(generate_sentiment, recommendations.index)

    ## Generate top 5 recommendations and send as Response
    top_recommendations = pd.DataFrame(results, columns=['Product', 'Sentiment Rate']).sort_values(by='Sentiment Rate', ascending=False)[:5]
    
    ## Set list of recommendations and pass it to webpage
    recommend_products = list(zip(top_recommendations['Product'], top_recommendations['Sentiment Rate']))
    return recommend_products