# TensorFlow
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# nltk package
import nltk
from nltk.stem import LancasterStemmer, SnowballStemmer, RegexpStemmer, WordNetLemmatizer 
#this was part of the NLP notebook
nltk.download('punkt')

# import sentence tokenizer
from nltk import sent_tokenize
# import word tokenizer
from nltk import word_tokenize
# list of stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# string and regex for text manipulation
import string
import re

# import lemmatizer
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# managing data and visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn key features
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split

# machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# miscellaneous
import emoji
import random
from random import randint
import pickle

import streamlit as st

def main():

    # importing models, tfdif vectorizer, and tokenizer
    dt = pickle.load(open("models/dt.pickle", "rb"))
    lr = pickle.load(open("models/lr.pickle", "rb"))
    lgb = pickle.load(open("models/lgb.pickle", "rb"))
    xgb = pickle.load(open("models/xgb.pickle", "rb"))
    Tfidf = pickle.load(open("tfidf_token/Tfidf.pickle", "rb"))
    tokenizer = pickle.load(open("tfidf_token/tokenizer.pickle", "rb"))

    # creating deep learning model
    dl_nlp = tf.keras.Sequential([
        tf.keras.layers.Embedding(40000, 16, input_length=130),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512,return_sequences=True)),
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation='sigmoid')
    ])

    dl_nlp.load_weights(r'models/weights.h5')
    

    # load its saved weights

    # removing punctuation
    punct =[]
    punct += list(string.punctuation)
    punct += '’'
    punct.remove("'")
    def remove_punctuations(text):
        for punctuation in punct:
            text = text.replace(punctuation, ' ')
        return text

    # remove emoji

    def text_has_emoji(text):
        for character in text:
            if character in emoji.UNICODE_EMOJI:
                return True
        return False

    def deEmojify(inputString):
        return inputString.encode('ascii', 'ignore').decode('ascii')


    # define lemmatization function
    nltk.download('wordnet')
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    def lemmatize(text):
        temp = []
        for txt in text.split():
            txt = lemmatizer.lemmatize(txt)
        temp.append(txt)
        return ' '.join(temp)

    # define stop words removal function
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    # create our own stop words set
    custom_set = set()
    # since 'coronavirus' and 'covid' appears in all 3 tweets, let's remove those words
    custom_set.add('coronavirus')
    custom_set.add('covid')
    for stopword in ENGLISH_STOP_WORDS:
        custom_set.add(stopword)
    
    # define function to count top n-grams
    def most_words(sentiment, process_status, ngram, n):
        if process_status == False:
            text = data[data['Sentiment']==sentiment]['Original Tweet']
        else:
            text = data[data['Sentiment']==sentiment]['Processed Tweet']

        count = CountVectorizer(ngram_range=(ngram, ngram)).fit(text)
        vocab = count.transform(text)
        word_sum = vocab.sum(axis = 0)
        freq = [(word, word_sum[0, index]) for word, index in count.vocabulary_.items()]
        freq = sorted(freq, key = lambda x: x[1], reverse=True)
        df_most_words = pd.DataFrame(freq[:n], columns = ['Words', 'Amount'])

        st.write(df_most_words)
        fig, ax = plt.subplots()
        ax = sns.barplot(x = 'Words', y = 'Amount', data=df_most_words)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=35)
        st.pyplot(fig)


    # define preprocessing function
    def preprocess_tweet(tweet):
        def lemmatize(text):
            temp = []
            for txt in text.split():
                txt = lemmatizer.lemmatize(txt)
                temp.append(txt)
            return ' '.join(temp)
        input = list()
        input.append(tweet)
        df_input = pd.DataFrame()
        df_input['OriginalTweet'] = input
        # lowercase the text
        df_input['cleaned_tweet'] = df_input['OriginalTweet'].apply(lambda x: x.lower())
        # getting rid of whitespaces
        df_input['cleaned_tweet'] = df_input['cleaned_tweet'].apply(lambda x: x.replace('\n', ' '))
        # remove links
        df_input['cleaned_tweet'] = df_input['cleaned_tweet'].str.replace(r'http\S+|www.\S+', '', case=False)
        # removing '>'
        df_input['cleaned_tweet'] = df_input['cleaned_tweet'].apply(lambda x: x.replace('&gt;', ''))
        # removing '<'
        df_input['cleaned_tweet'] = df_input['cleaned_tweet'].apply(lambda x: x.replace('&lt;', ''))
        # checking emoji
        df_input['emoji'] = df_input['cleaned_tweet'].apply(lambda x: text_has_emoji(x))
        # remove emoji
        df_input['cleaned_tweet'] = df_input['cleaned_tweet'].apply(lambda x: deEmojify(x))
        # remove punctuation
        df_input['cleaned_tweet'] = df_input['cleaned_tweet'].apply(remove_punctuations)
        # remove ' s ' that was created after removing punctuations
        df_input['cleaned_tweet'] = df_input['cleaned_tweet'].apply(lambda x: str(x).replace(" s ", " "))
        # apply lemmatization
        df_input['lemmatized'] = df_input['cleaned_tweet'].apply(lambda x: lemmatize(x))
        # After lemmatization, some pronouns are changed to '-PRON' and we're going to remove it
        df_input['lemmatized'] = df_input['lemmatized'].apply(lambda x: x.replace('-PRON-', ' '))
        # apply tokenization
        df_input['tokenized_tweet'] = df_input['lemmatized'].apply(word_tokenize)
        # remove stop words
        df_input['tokenized_tweet'] = df_input['tokenized_tweet'].apply(lambda word_list: [x for x in word_list if x not in custom_set])
        # remove numbers
        df_input['tokenized_tweet'] = df_input['tokenized_tweet'].apply(lambda list_data: [x for x in list_data if x.isalpha()])
        # finalizing the preprocessing
        finale = list()
        for list_of_words in df_input.tokenized_tweet:
            finale.append(' '.join(list_of_words))

        df_input['final_text'] = finale

        return df_input.final_text

    # define tf-idf function (for machine learning algorithms)
    def tfidf_text(tfidf_model, text):
        tfidf_new = TfidfVectorizer(ngram_range=(1,2), 
                        vocabulary = tfidf_model.vocabulary_)
        vector = tfidf_new.fit_transform(text)
        return vector

    # define padding function (for deep learning algorithm)
    def padding(tokenizer_model, text):
        sequence = tokenizer_model.texts_to_sequences(text)
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=130)
        return padded

    def predict_proba(model, vector, print_all = False, deep_learning = False):
        if deep_learning == False:
            array = model.predict_proba(vector)
            prediction = model.predict(vector)[0]
        else:
            array = model.predict(vector)
            prediction = np.argmax(array)

        neg_proba = array[0][0]
        neu_proba = array[0][1]
        pos_proba = array[0][2]

        if print_all == True:
            print('Positive Sentiment Probability: ', round(pos_proba*100,3), '%')
            print('Neutral Sentiment Probability: ', round(neu_proba*100, 3), '%')
            print('Negative Sentiment Probability: ', round(neg_proba*100, 3), '%')

        if prediction == 0:
            st.write("This model predicts that your tweet's setiment is: Negative.")
            st.write('Prediction probability:')
        elif prediction == 1:
            st.write("This model predicts that your tweet's setiment is: Neutral.")
            st.write('Prediction probability:')
        else:
            st.write("This model predicts that your tweet's setiment is: Positive.")
            st.write('Prediction probability:')
  
        output = {'Sentiment':['Positive', 'Neutral', 'Negative'],
            'Probability':[pos_proba, neu_proba, neg_proba]}
        df_output = pd.DataFrame(output, columns = ['Sentiment', 'Probability'])

        fig, ax = plt.subplots()
        ax = sns.barplot(x = 'Sentiment', y = 'Probability',data = df_output)
        s = [pos_proba, neu_proba, neg_proba]

        for i,p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                height + 0.003, str(round(s[i]*100,2))+'%',
                ha="center") 

        st.pyplot(fig)

        return df_output

    st.sidebar.title('Navigation')
    pages = st.sidebar.radio("Pages", ("Home Page", "Tweet Sentiment Classifier", 
        "Summary", "About the Author"), index = 0)
    if pages == "Home Page":
        
        st.title('Welcome to COVID-19 Tweet Sentiment Classification Project!')
        st.image('twitter.png', width = 600)
        st.markdown("Please open the sidebar and navigate to any pages you'd like. Enjoy!")
    
    elif pages == "Tweet Sentiment Classifier":

        st.title('COVID-19 Tweet Sentiment Classifier')
        st.subheader('Premise')
        st.write("Four machine learning algorithms (Logistic Regression, Decision Tree, XGBoost Classifier, \
            and Light GBM Classifier) and a TensorFlow-Keras deep learning model were trained on a set of labeled tweets \
            about COVID-19 from a Kaggle dataset. Each tweet is labeled as having 'Positive', 'Neutral', or 'Negative' sentiment.")
        st.write("A train-test split evaluation indicates that our deep learning model performed the best.")
        st.write("Now, let's try to test these classifiers on your imaginary tweets.")
        st.write("(Tip: Try to write something with a sentiment in mind, and see if the classifiers guess the correct sentiment)")
        if st.button("Click here to see some interesting tweets"):
            st.markdown("Here are a few tweets with interesting outcomes (I won't spoil the result):")
            st.markdown("- my friend died because of this virus. unbelievable")
            st.markdown("- we're all angry at the egoistic people who tries to benefit from others during this time of need")
            st.markdown("- don't surrender! we can still fight till the end, and win against this pandemic")
            st.markdown("You might think that the first two are obviously negative tweets, \
                while the last one is a positive motivational tweet. However, some models tend to disagree :)")
            st.markdown("These tweets show us that a deep learning model with bidirectional LSTM layers are great at \
                sentiment analysis, and should perform better than classic ML algorithms.")
        st.subheader('Write something about COVID-19:')
        raw_tweet = st.text_area("Your tweet:", max_chars = 140)
        tweet = preprocess_tweet(raw_tweet)
        vector = tfidf_text(Tfidf, tweet)
        pad = padding(tokenizer, tweet)

        if st.button('Classify the sentiment of the Tweet'):
            st.subheader('Deep Learning Model with Bi-Directional LSTM')
            predict_proba(dl_nlp, pad, deep_learning = True)
            st.subheader('Logistic Regression')
            predict_proba(lr, vector)
            st.subheader('Light GBM Classifier')
            predict_proba(lgb, vector)        
            
    elif pages == "Summary":
        data = pd.read_csv('dataset/Corona_NLP_Finalized_Dataset.csv')
        st.title('Summary and Insights')
        st.subheader('Dataset')
        display = st.checkbox("Display Data", False)
        if display == True:
            st.write(data)
        else:
            pass

        st.subheader('Brief Summary')
        st.markdown('The raw Kaggle dataset is first cleaned, tokenized, and vectorized before being fed into ML/DL algorithm to train models.')
        st.markdown('There are 3 classes: Positive, Neutral, and Negative emotion(s).')
        st.markdown('There are 3 models deployed: Logistic Regression, Light GBM Classifier, and a Deep Learning model with Bidirectional LSTM layer.')
        st.markdown('The best performing model is the Deep Learning model, with 0.83 accuracy on test set'.)
        st.subheader('Process')
        st.markdown('The dataset is taken from [this](https://www.kaggle.com/datatattle/covid-19-nlp-text-classification) Kaggle dataset.')
        st.markdown("In total (combining both train and test csv files) there are 44,955 tweets scraped from Twitter throughout 2020. \
            These tweets are talking about the COVID-19 pandemic.") 
        st.markdown("The original uploader of the dataset has labeled the sentiments of each tweet into \
            five: 'Extremely Positive', 'Positive', 'Neutral', 'Negative', 'Extremely Negative'. However, in this exercise, we're combining 'Extremely Positive' \
            with 'Positive', and 'Extremely Negative' with 'Negative'. ")
        st.markdown("This is the distribution of the number of tweets:")

        fig, ax = plt.subplots()
        ax = sns.countplot(x = 'Sentiment', data = data)
        st.pyplot(fig)

        st.subheader('Preprocessing Method')
        st.markdown("1. Setting all letters to lowercase") 
        st.markdown("2. Removing whitespace")
        st.markdown("3. Removing links")
        st.markdown("4. Removing '<>'")
        st.markdown("5. Removing emoji")
        st.markdown("6. Removing punctuation")
        st.markdown("7. Lemmatization using WordNet (from nltk)")
        st.markdown("8. Tokenization")
        st.markdown("9. Removing stop words (from sklearn's list)")
        st.markdown("10. Removing numbers")
        st.markdown("11. Applying TF-IDF Vectorization (for ML model) or TensorFlow sequence padding (for deep learning model)")
        st.subheader('Top N-grams of Processed Tweets')

        st.markdown("After texts are processed, we have the term n-gram in NLP. N-gram is a continuous sequence of 'n' unit of speech \
            (syllables, letters, words, etc). The unit of speech usually chosen is words.")
        st.markdown("So, a 1-gram (often called unigram) is one word. A 2-gram (usually called bigram) are two words that appear consecutively.")
        st.markdown("For example, if there are 20 bigrams of 'corona crisis', it means that the phrase 'corona crisis' appears 20 times in our text.")
        st.markdown("Each sentiment has its own unique rank of top n-grams. Select the options below to check 'em out.")    
        sentimen = st.selectbox("Sentiment:", ("Positive", "Neutral", "Negative"))
        n_gram = st.selectbox("N-gram:", (1, 2))
        top = st.selectbox("How many top n-grams?", (8, 9, 10, 11, 12, 13, 14, 15))
        if st.button("View ranking of top n-grams"):
            most_words(sentimen, True, n_gram, top)

        st.subheader('ML Model Training and Evaluation')
        st.markdown('Each model is trained on the train set, and then tested on test set. \
            The train-test split is randomized by sklearn (random_state = 42), and the ratio of the test split is 0.15')

        st.markdown('Training and evaluation on different train-test splits, as well as different random state \
            might give slightly different result.')
        
        st.markdown('Performance of ML Models:')
        st.markdown('- Logistic Regression with accuracy of 0.78')
        st.markdown('- Light GBM Classifier with accuracy of 0.78')
        st.markdown('To see the detailed classification report (f1 score, recall, precision), \
            please look at the google colab/jupyter notebook.')

        st.subheader('Deep Learning Model and Evaluation')
        st.markdown('The deep learning model gives us the highest accuracy, which is around 0.83')
        st.markdown("This is the model architecture:")
        st.markdown("- Embedding layer with 40000 of vocab list, and 16 embedding dimensions")
        st.markdown("- Batch Normalization layer")
        st.markdown("- Bidirectional LSTM with 512 units")
        st.markdown("- GlobalMaxPool1D")
        st.markdown("- Dropout layer (0.2)")
        st.markdown("- Dense layer with 64 units and 'relu' activation function")
        st.markdown("- Dropout layer (0.2)")
        st.markdown("- Dense layer with 3 units (for output) with 'sigmoid' activation function")
        st.markdown("- Model is compiled with 'sparse categorical crossentropy' loss function and \
            'nadam' optimizer.")
        st.markdown("- Model is trained in only 2 epochs. It overfits if we give it more epochs")
        st.subheader('About Bidirectional LSTM')
        st.markdown('Our result shows that the deep learning model outperforms all classic ML models. \
            This is because we have a Bidirectional LSTM layer.')
        st.markdown("An LSTM layer (Long Short Term Memory) is a Recurrent Neural Network layer. \
            This allows our neural network to learn the sequence of our words.")
        st.markdown("Instead of only \
            counting the interval and/or number of words (this is what classic ML models do), \
            LSTM learns a tweet just like how we, humans, understand a sentence by reading it from \
            start to finish. The word before influences the meaning of the word next to it.")
        st.markdown("However, regular RNN tends to 'forget' words that are 'far away'. For example, when it learns \
            the meaning of the 7th word from the context given by the 6th word, it does not associates this '7th' \
            word to the 1st word. This is where LSTM comes in.")
        st.markdown("LSTM makes the short-term memory of a neural network...longer. It allows model to still remember the 'context' \
            of the 1st word while it is reading the 7th, 8th, etc words.")
        st.markdown("By making it Bidirectional, it also reads the sentence backwards. Now, the last words will carry on its meaning \
            in contextualizing the first words. This allows our model to learn the sentence structure even better.")
        st.subheader('Takeaway')
        st.markdown('- This project shows that training a Bidirectional LSTM Neural Network is a good way to tackle NLP Sentiment Classification problems.')
        st.markdown('- We need to preprocess texts properly before it can be inputed to our models. Preprocessing different \
            style of texts require its own unique approach. For example, our preprocessing steps here might only be useful for \
            twitter-type texts.')
        st.markdown('- Three packages which helps us in preprocessing texts: nltk, spaCy, scikit-learn.')
        st.markdown('- After importing a list of stop words, we can customize it by adding our own words there.')
        st.markdown('- Although performing the best, training a deep learning model takes more time and computational power.')
        st.markdown('Ways to improve/enrich this project/similar projects:')
        st.markdown('- Use more data, preferrably over 100k tweets.')
        st.markdown('- Apply other methods of word embeddings, such as Word2Vec, Bag-of-Words, etc')
        st.markdown('- Try other order/combination of layers in the deep learning model.')


    
    elif pages == "About the Author":
        st.title('About the Author')
        st.subheader("Greetings! My name is Grady Matthias Oktavian. Nice to meet you!")
        st.write("I graduated at 2020 from Universitas Pelita Harapan with the title Bachelor of Mathematics, majoring in Actuarial Science. \
            Currently, I'm studying at Pradita University as a student in the Master of Information Technology degree majoring in Big Data and IoT. \
            I am also employed as an IFRS 17 Actuary at PT Asuransi Adira Dinamika.") 
        st.write("I like learning about statistics and mathematics. Since today is the age of Big Data, I find that most people who \
            aren't majoring in mathematics might find themselves overwhelmed with large amounts of data. My dream is to help people make better \
            decision through a data-driven approach.") 
        st.write("In order to do that, I am happy to wrangle and analyze raw data, creating models based on it, and \
            conveying insights I gained to others who are not well-versed in data, so they can understand it without having to \
            get their hands dirty with the data. I hope through my help, people can understand things better, busineess owners can \
            make better contingency plans and take better decisions.")
        st.markdown("Currently, I am certified by Google as a [TensorFlow Developer](https://www.credential.net/794f2bb6-d377-4b5b-ac9d-9d3bed582d2d), \
            and a [Cloud Professional Data Engineer](https://www.credential.net/df7c3d9d-011a-41fd-9d64-49ada5a0619c#gs.ktrahi).")
        st.write("If you wish to have a suggestion for this project, or contact me for further corresnpondences, \
            please reach out to me via email at gradyoktavian@gmail.com, or send me a message to \
            my [LinkedIn profile](https://www.linkedin.com/in/gradyoktavian/).")
        st.write("Thank you! Have a nice day.")


if __name__ == '__main__':
    main()


