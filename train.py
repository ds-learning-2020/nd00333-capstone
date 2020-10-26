from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
import sys
from pip._internal import main as pipmain
pipmain(["install", "nltk"])
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 





#df = TabularDatasetFactory.from_delimited_files("Corona_NLP_train.csv").to_pandas_dataframe(encoding='latin1')
df = pd.read_csv("Corona_NLP_train.csv",encoding='latin1')

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.stemmer = SnowballStemmer("english")
    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in word_tokenize(doc) if t.isalpha()]

def clean_data(inp_df):
    df_modified = inp_df[["OriginalTweet", "Sentiment"]].copy()
    df_modified.dropna(inplace=True)
    df_modified["OriginalTweet"] = df_modified["OriginalTweet"].str.lower().str.replace("[^a-zA-Z]","")
    count_vec = TfidfVectorizer(tokenizer = LemmaTokenizer(), token_pattern = None, ngram_range= (1, 2))
    
    X = count_vec.fit_transform(df_modified["OriginalTweet"])



    df_modified["Sentiment"] = df_modified["Sentiment"].astype("str")
    df_modified = df_modified[df_modified["Sentiment"] != "nan"]


        
    positive = df_modified["Sentiment"].isin(["Neutral", "Positive", 'Extremely Positive'])
    negative = df_modified["Sentiment"].isin(['Extremely Negative', 'Negative'])

    df_modified.loc[positive, "Sentiment"] = 1
    df_modified.loc[negative, "Sentiment"] = 0

    df_modified["Sentiment"] = df_modified["Sentiment"].astype("int")

    labels = df_modified["Sentiment"]
    

    return X, labels

x, y = clean_data(df)

run = Run.get_context()
    


x_train, x_test , y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

#def main():
# Add arguments to script
parser = argparse.ArgumentParser()

parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

args = parser.parse_args()

run.log("Regularization Strength:", np.float(args.C))
run.log("Max iterations:", np.int(args.max_iter))

model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
run.log("Accuracy", np.float(accuracy))


#main()
