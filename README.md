
# Sentiment Analysis on Coronavirus related Tweets

This project aims to conduct a sentiment analsysis on data collected from Twitter. Classification algorithms are used on the data to predict whether a tweet is positive or negative.

## Dataset

### Overview

The data was obtained from a dataset uploaded on Kaggle - [Coronavirus tweets NLP - Text Classification](https://www.kaggle.com/datatattle/covid-19-nlp-text-classification). It consists of the following columns -

- UserName
- ScreenName
- Location
- TweetAt
- OriginalTweet
- Sentiment

The dataset consists of 41159 entries.

As the names suggest, OriginalTweet consists of the tweets themselves (text), and the Sentiment column consists of 6 labels -

- Extremely Negative
- Negative
- Neutral
- Positive
- Extremely Positive


#### Data preprocessing

I performed the following steps to preprocess the data. These can be found in the `clean_data()` function in `train.py` as well.

- I have only used the OriginalTweet and Sentiment columns.
- Removed any null values
- To simplify the approach, I converted the Extremely Negative and Negative labels to a `0`. And the remaining labels - Neutral, Positive and Extremely Positive, I converted those to a 1.
- I turned the tweets into lowercase and removed any non-alphabetic characters

There are some additional steps I took in preprocessing the data for the model relying on Hyperdrive -

- I tokenized the individual words in the tweet using `nltk`'s `word_tokenize()` method.
- I stemmed the tokenized words using the `SnowballStemmer()`.
- Using `TfidfVectorizer()` I generated frequency scores for each stemmed+tokenized word.

These frequency scores were then passed in as data to the model for classification.

These additional steps don't apply to the AutoML approach.

### Task
I am going to train a classifier that predicts the sentiment of a tweet, whether it is positive or negatie. I will only be working with the actual tweet as feature for this project to predict the sentiment.

### Access
CSV formatting can change depending on where the file was downloaded to. As a result, when I try to register the dataset in a Azure datastore, the data gets mixed up between columns. I couldn't find a fix to this through Azure (inspite of trying out all possible changes when registering it). 

To mitigate that, I directly uploaded the dataset csv (through Notebooks -> Upload files) and used pandas to read the csv into the appropriate files (`train.py` and `automl.ipynb`). The dataset was downloaded from Kaggle first on a local system, then added to my project's GitHub repo, and then downloaded from the repo onto the VM's Windows system.

Since the rubrics mention -

> Note: The exact method of uploading and accessing the data is not important.

I don't think this approach should be an issue.

## Automated ML
These are the settings and config I used for the AutoML -
```
automl_settings = {"experiment_timeout_minutes":30,
    "task":"classification",
    "primary_metric":"accuracy",
    "training_data":df_modified,
    "label_column_name":"Sentiment",
    "n_cross_validations":3}
```

The experiment timeout was set to 30 minutes. Since we have a classification task (predicting the Sentiment), accuracy was chosen as a metric that AutoML optimizes on. Because of the size of the dataset, I kept the number of cross validation folds to 3.


### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
I used Logistic Regression as the model for this experiment. Since my classification task involved only two classes (positive or negative sentiment) Logistic Regression was a good starting point to experiment with because it models the probabilities for problems that havev two possible outcomes (our two classes). The probabilities returned by the model can also help us understand how confident the model is of its prediction.

I used only two parameters for this -
1. `C` - This is the inverse of the regularization strength that helps define how strong we want our regularization to be. Higher the value of `C`, weaker the regularization.
2. `max_iter` - This is the maximum number of iterations for the model to converge.


For both of the above parameters I set the following ranges for the hyperparameter search - 

1. `C` - A uniform distribution over the range (0.1, 0.9)
2.`max_iter` - 4 choices of - (25, 50, 75, 100)

The above have been defined in the `hyperparameter_tuning.ipynb` Notebook as well.

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response
