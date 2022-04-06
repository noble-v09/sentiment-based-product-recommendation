from flask import Flask, redirect, request, url_for, render_template, flash
import pandas as pd
import numpy as np
import pickle
# import swifter
from preprocessor import *

## Import model
with open('./models/model.pkl', 'rb') as f:
    model = pickle.load(f)

## Import recommendations matrix
with open('./models/user_recommendations.pkl', 'rb') as f:
    user_ratings = pickle.load(f)

## Import tfidf vectorizer
with open('./models/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

## Get user_list
user_list = list(user_ratings.index.unique())
## Prepending --SELECT-- as first option
user_list.insert(0, "--SELECT--")

## Read reviews
reviews = pd.read_csv('./data/sample30.csv', usecols=['name', 'reviews_text'])

# print(reviews['reviews_text'][100])
# print(nltk.__version__)
# reviews['reviews_text'] = reviews['reviews_text'].swifter.apply(lemmatize_text)

app = Flask(__name__)
app.config["SECRET_KEY"] = "development"

@app.route("/")
def index():
    return render_template("index.html", user_list=user_list, user=None, show_recommendations=False, recommend_products=list())

@app.route("/recommend", methods=['GET', 'POST'])
async def recommend():
    user = request.form.get("user_name")

    if user==None or user=="--SELECT--":
        flash("Please select a user from the list", "danger")
        return redirect(url_for("index"))
    ## Get top 20 recommendations    
    recommendations = user_ratings.loc[user].sort_values(ascending=False)[:20]

    sentiment_rates = pd.DataFrame(columns=['Product', 'Sentiment Rate'])
    ## Get reviews for each recommendations
    for idx, product in enumerate(recommendations.index):
        prod_reviews = reviews.loc[reviews['name']==product, ['reviews_text']]
        ## Text Preprocessing
        ## Ideally, text preprocessing is ideally done in two ways:
        ## Case 1: We do the preprocessing step directly in DB/File itself and keep it ready, rather than doing it at runtime
        ## Case 2: App Startup
        ## Case 3: At runtime, as soon as api is called and the data is read from DB/file
        ## Here, due to resource constraints in Heroku, we are doing it at runtime
        prod_reviews['reviews_text'] = preprocess_text(prod_reviews['reviews_text'])
        prod_reviews['reviews_text'] = prod_reviews['reviews_text'].apply(lemmatize_text)
        
        ## Apply TFIDF Vectorizer to reviews
        X = pd.DataFrame(data=tfidf.transform(prod_reviews['reviews_text']).toarray(), columns=tfidf.get_feature_names())

        ## Predict Sentiment --> Negative == 0, Positive == 1
        y = model.predict(X)

        ## Append final sentiment rate for each product
        sentiment_rates = sentiment_rates.append(pd.Series([product, round(np.sum(y)/len(y)*100, 2)], index=sentiment_rates.columns), ignore_index=True)
    
    # print(sentiment_rates.sort_values(by='Sentiment Rate', ascending=False))
    ## Generate top 5 recommendations and send as Response
    top_recommendations = sentiment_rates.sort_values(by='Sentiment Rate', ascending=False)[:5]
    
    ## Set list of recommendations and pass it to webpage
    recommend_products = list(zip(top_recommendations['Product'], top_recommendations['Sentiment Rate']))
    return render_template("index.html", user_list=user_list, user=user, show_recommendations=True, recommend_products=recommend_products)


if __name__ == '__main__':
    # app.run(debug=True)
    # app.run(host="0.0.0.0", port=5000)
    app.run()


