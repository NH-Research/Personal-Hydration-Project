import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import kagglehub
import os

path = kagglehub.dataset_download("sonalshinde123/daily-water-intake-and-hydration-patterns-dataset")

# print("Path to dataset files:", path)


directory_listing = os.listdir(path) # get a list of the directories in that path

# print(directory_listing) # Print the directory list

csv_directory = directory_listing[0] # grab the first index (the csv) as there is only one file - will use as data

csv_path = os.path.join(path, f"{csv_directory}") # use the os module and use path to get the entries full path name and use join to 

# print(csv_path)

csv = pd.read_csv(csv_path)

# print(csv.head)

# print(csv.shape)


column_names = list(csv.columns)

# print(column_names)

input_headers = column_names[0:6]

# print(input_headers)

target_header = column_names[6]

# print(target_header)

# print(csv.iloc[0,6])

raw_target = csv.iloc[:,6]   #use iloc in pandas and straight x[] in numpy


# print(f"\nThe title of the target data is: {raw_target}")

raw_input = csv.iloc[:,0:6]

# print(f"\n\nThe titles of the input data are: {raw_input.iloc[0,:]}")

# print(raw_target.name) # Name of the csv column no longer stored when printed or when you do .head -> now stored as an attribute: ".name" 

####        Now we have managed to extract the data heading and seperate the input to output data --- move to cleaning

# We want to remove duplicate rows, NaNs and any infs. 
# Additionally, we want to one hot encode the data, as it is a classification problem.

# To one hot encode SOME of the input data, we must first establish what data must be encoded (non-continuous)
# Will use LabelBinarizer for y data as it is either good/bad and isnt multi-class

###     Cool easy way of getting lables - use label map in pandas funcs

# label_map = {"poor": 0, "good": 1}
# y = df["Hydration_Status"].map(label_map)


###     LabelBinarizer way:

# bin = LabelBinarizer()
# Y_encoded = bin.fit(raw_target)
# Y_encoded_test = bin.fit_transform(raw_target)
# print(f"\n \n The binarizer when just using '.fit': \n{Y_encoded}")
# print(f"\n \n The binarizer when using '.fit_transform': \n{Y_encoded_test}")


###### As our input data is mixed in discrete and ocntinuous data, we must identify what columns are what

def cnv2_pd(x):
    x = pd.DataFrame(x)
    return x

def unique_data(data): 

    x = cnv2_pd(data)
    num_unique_list = []
    for i in x:
        num_unique_list = x.nunique



    return num_unique_list



enc = OneHotEncoder()
X_encoded = enc.fit_transform()



# x_np = 

# y_np = 