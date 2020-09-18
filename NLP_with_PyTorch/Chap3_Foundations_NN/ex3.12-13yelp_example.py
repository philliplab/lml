import collections
import numpy as np
import pandas as pd
import re

from argparse import Namespace

base_dir = "/home/phillipl/0_para/3_resources/PyTorch/yelp_review_polarity_csv"

args = Namespace(
    raw_train_dataset_csv = base_dir + "/train.csv",
    raw_test_dataset_csv = base_dir + "/test.csv",
    proportion_subset_of_train = 0.1,
    train_proportion = 0.7,
    val_proportion = 0.15,
    test_proportion = 0.15,
    output_munged_csv = base_dir + "/reviews_with_splits_lite.csv",
    seed = 1337
)

by_rating = collections.defaultdict(list)

train_reviews = pd.read_csv(args.raw_train_dataset_csv, header = None, names = ['rating', 'review'])

for _, row in train_reviews.iterrows():
    by_rating[row.rating].append(row.to_dict())

train_reviews.iloc[0:10,:]

train_reviews.rating = train_reviews.rating.apply({1: 'negative', 2: 'positive'}.get)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1", text)
    text = re.sub(r"[^a-zA-Z.,!?]", r" ", text)
    return text

train_reviews.review.apply(preprocess_text)
train_reviews = train_reviews.assign(old_review = train_reviews['review'])

train_reviews.review = train_reviews.review.apply(preprocess_text)

train_reviews = train_reviews.assign(split = ['val' if i < 0.1 else 'test' if i > 0.9 else 'train' for i in np.random.random(train_reviews.shape[0])])

train_reviews.split.value_counts()

train_reviews.iloc[0:50000, [0, 1, 3]]

train_reviews.iloc[0:50000, [0, 1, 3]].to_csv(args.output_munged_csv, index=False)









