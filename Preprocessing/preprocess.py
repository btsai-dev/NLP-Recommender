import gzip
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import csv
from html.parser import HTMLParser
from io import StringIO
import re
import pandas as pd
import random
from random import sample

def parse(path):
    with gzip.open(path, mode="rt") as f:
        data = [json.loads(line) for line in f]
    return data

def load_data(path):
    data = []
    for d in parse(path):
        data.append(d)
    return data

# strip HTML tags and entities
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(text):
    text = re.sub('(\&lt\;).*?(\&gt\;)', '', text)
    s = MLStripper()
    s.feed(text)
    return s.get_data()

data = load_data('Video_Games_5.json.gz')
output_name = 'output.csv'

user_reviews = dict()
product_set = set()
for review in tqdm(data, desc="Parsing reviews"):
    user_id = review.get('reviewerID', None)
    product_id = review.get('asin', None)
    rating = review.get('overall', None)
    text = strip_tags(review.get('reviewText', ''))

    if len(text) <= 0 or user_id is None or product_id is None or rating is None:
        continue

    if product_id not in product_set:
        product_set.add(product_id)
        
    if user_id not in user_reviews:
        user_reviews[user_id] = []
    user_reviews[user_id].append({"ProductID": product_id, "Rating": rating, "Review": text})

# Sort product list and convert to mapping
product_list = sorted(list(product_set))
product_mapping = {product_id: index for product_id, index in zip(product_list, range(0, len(product_set)))}

dataframe_result = list()
dropped = 0

# Median number or reviews per user is six.
for user_id, reviews in tqdm(user_reviews.items(), desc="Creating user data"):
    # Need at least 6 reviews to be acceptable
    if len(reviews) < 6:
        print("Error: Not enough reviews for " + user_id + ", (", len(reviews), ")")
        continue

    list_of_product_encodings = []
    dupe_products = set()
    for review_dict in reviews:
        product_id = review_dict["ProductID"]
        rating_num = review_dict["Rating"]
        review_text = review_dict["Review"]
        
        product_index = product_mapping[product_id]
        encoding_dict = dict()
        review_encoded = np.zeros(len(product_set), dtype=np.int8)
        review_encoded[product_index] = 1
        encoding_dict["ProductID"] = product_id
        encoding_dict["ReviewEncoded"] = review_encoded
        encoding_dict["ProductRating"] = rating_num
        encoding_dict["ReviewText"] = review_text
        # if product_id not in dupe_products:
        #     dupe_products.add(product_id)
        # else:
        #     print("Warning: UserID", user_id, "has multiple product reviews for", product_id,".")
        
        list_of_product_encodings.append(encoding_dict)
  
    # List of data tuples
    all_review_data = list()

    for holdout_review_encoding_dict in list_of_product_encodings:
        # Now loop through all, excluding holdout, and add to list
        other_review_productid_list = []
        other_review_rating_list = []
        other_review_text_list = []
        
        for other_review_encoding_dict in list_of_product_encodings:
            other_product_id = other_review_encoding_dict["ProductID"]
            other_review_encoding = other_review_encoding_dict["ReviewEncoded"]
            other_rating_num = other_review_encoding_dict["ProductRating"]
            other_review_text = other_review_encoding_dict["ReviewText"]
            
            if other_product_id == holdout_review_encoding_dict["ProductID"] and other_review_text == holdout_review_encoding_dict["ReviewText"]:
                continue

            other_review_productid_list.append(other_product_id)
            other_review_rating_list.append(other_rating_num)
            other_review_text_list.append(other_review_text)

        for i in range(len(other_review_productid_list)):
            product_tuple = (other_review_productid_list[i], other_review_rating_list[i], other_review_text_list[i])
            if product_tuple not in all_review_data:
                all_review_data.append(product_tuple)

        # Stuff may have been filtered out for being fully dupes. Remove if number of reviews is less than 6.
    if len(all_review_data) < 6:
        print("Error: Not enough reviews for " + user_id + ", (" + str(len(all_review_data)) + ")")
        dropped += 1
        continue
        
    row_data = dict()
    row_data["user_id"] = user_id
    row_data["review_data"] = all_review_data
    row_data["review_count"] = len(all_review_data)
        
    dataframe_result.append(row_data)
        
dataframe_df = pd.DataFrame(dataframe_result)
print("Dropped", dropped, "users for having too few reviews!")
# Chunkify, then save iteratively

print("Saving data to csv: ", output_name)
dataframe_df.to_csv(output_name, index=None)
print("Save completed.")