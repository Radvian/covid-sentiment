# Welcome to covid-sentiment!

In this project, we import data from https://www.kaggle.com/gpreda/covid19-tweets and train a few machine learning and deep learning models to do a sentiment analysis on labeled COVID-19 tweets. Then, we deploy these models to a web app on Streamlit so we can write our own tweets about COVID-19 and see our tweet's sentiments according to our trained models.

# Model Description

Raw tweets are preprocessed first to remove links, punctuations, and stop words. Then, it is vectorized into words, and lemmatized. Afterwards, using Tf-idf, each tweet is converted into vectors of numbers which are then processed with ML Algorithms. There are 4 machine learning model used: Logistic Regression, XGBoost, Light GBM, and Decision Tree. 

For the deep learning model, we use embedding layer from TensorFlow before we create and train a model with Bidirectional LSTM layers. The result shows that the deep learning model achieves higher accuracy compared to our machine learning models. 

Finally, we deploy the trained models on a web app on streamlit in which users can type their own 'tweets' about COVID-19 and let our models classify the sentiment. 

# Web App

To run the web app locally, you can download this repository and type ```streamlit run streamlit-sentiment.py``` in the folder's terminal. 
To view the hosted web app, you can visit the following link: https://share.streamlit.io/radvian/covid-sentiment/main/streamlit-sentiment.py
