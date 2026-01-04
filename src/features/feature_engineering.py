import numpy as np
import pandas as pd
import os

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

import yaml

import logging

## Create and configure logger
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> int:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    max_features = params['feature_engineering']['max_features']
    return max_features

def load_processed_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        logger.error('File not found')
        raise
    except pd.errors.ParserError as e:
        logger.error('CSV parsing error')
        raise
    except Exception as e:
        logger.error('some error occured')
        raise
    
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        # data_path = os.path.join(data_path, 'processed')
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.")
        print(e)
        raise   
    
def apply_bow_vectorization(X_train,X_test, max_features):
    vectorizer = CountVectorizer(max_features=max_features)
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    return X_train_bow, X_test_bow

def main():
    
    max_features=load_params('params.yaml')
    
    # Load processed data
    data_path = os.path.join("data","processed")
    
    train_data = load_processed_data(os.path.join(data_path, 'train_processed.csv'))
    test_data = load_processed_data(os.path.join(data_path, 'test_processed.csv'))
    
    train_data.fillna('',inplace=True)
    test_data.fillna('',inplace=True)

    X_train = train_data['content'].values
    y_train = train_data['sentiment'].values

    X_test = test_data['content'].values
    y_test = test_data['sentiment'].values
    
    X_train_bow, X_test_bow = apply_bow_vectorization(X_train,X_test, max_features)

    train_df = pd.DataFrame(X_train_bow.toarray())
    train_df['label'] = y_train

    test_df = pd.DataFrame(X_test_bow.toarray())
    test_df['label'] = y_test

    # store the data inside features folder
    data_path = os.path.join("data","features")
    save_data(train_df, test_df, data_path)
    
if __name__ == '__main__':
    main()


