# Sentiment Based Product Recomendation System

In this project, we have used User-User based Collaborative Filtering Recommendation system for custom recommendation of products for all users. 
Further, sentiment analysis is performed on reviews of those products recommended by the system and then top 5 best custom recommendations having high positive sentiments are displayed for each user. 
From the given list of users, please select one to view their custom product recommendations!

For training, we have build two types of Collaborative Filtering Recommendation system for given dataset

* User-User based Recommendation system
* Item-Item based Recommendation system

We then evaluated both and picked the best

We then proceed to build sentiment analysis models using different algorithms such as Logistic Regression, Random Forest, XGBoost etc. based on product reviews to increase efficiency of our recommendation system by providing top custom recommendations for each user.

The final project can be viewed here at this link: https://sbprs-demo-nv.herokuapp.com/

For Heroku Deployment; due to resource constraint (constantly getting "R14: Memory quota exceeded" error), have directly used clean dataset consisting of 2 columns --> pre-proccessed, lemmatized reviews text and product names only, to avoid memory exceeded issues. During recommendation by Web API, TFIDF is applied on clean reviews for top 20 recommendations and then Sentiment Classification is done to generate top 5 best recommendations