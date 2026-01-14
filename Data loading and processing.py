import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import kagglehub
import os

path = kagglehub.dataset_download("sonalshinde123/daily-water-intake-and-hydration-patterns-dataset")

print("Path to dataset files:", path)


directory_listing = os.listdir(path) # get a list of the directories in that path

print(directory_listing) # Print the directory list

csv_directory = directory_listing[0] # grab the first index (the csv) as there is only one file - will use as data

csv_path = os.path.join(path, f"{csv_directory}") # use the os module and use path to get the entries full path name and use join to 

print(csv_path)

csv = pd.read_csv(csv_path)

print(csv.shape)

raw_target = csv.iloc[:,6]   #use iloc in pandas and straight x[] in numpy

print(raw_target.iloc[0])

# raw_input = csv.iloc[:,0:6]