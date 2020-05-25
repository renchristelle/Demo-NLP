# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

import re
import os
import tarfile
import requests

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Input Folder
imdb_folder = dataiku.Folder("khqULen5")
dirname = imdb_folder.get_path()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Reading the reviews into a Folder

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Load all files from a directory in a DataFrame
def load_directory_data(directory):
    data = {
        "text": [],
        "sentiment": []
    }
    for file_path in os.listdir(directory):
        with open(os.path.join(directory, file_path), "r") as f:
            data["text"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1.).reset_index(drop=True)

# Load train and test reviews
train_df = load_dataset(os.path.join(dirname, "aclImdb", "train"))
train_df["sample"] = "train"
test_df = load_dataset(os.path.join(dirname, "aclImdb", "test"))
test_df["sample"] = "test"


train_df = train_df[['text', 'sentiment', 'polarity', 'sample']]
test_df = test_df[['text', 'sentiment', 'polarity', 'sample']]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
output_df = pd.concat([train_df, test_df])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
imdb_data = dataiku.Dataset("IMDB_data")
imdb_data.write_with_schema(output_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write Dataframe in Dataset
# imdb_train = dataiku.Dataset("IMDB_train")
# imdb_train.write_with_schema(train_df)
# imdb_test = dataiku.Dataset("IMDB_test")
# imdb_test.write_with_schema(test_df)