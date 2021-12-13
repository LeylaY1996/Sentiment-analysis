import pandas as pd
import numpy as np
import nltk
import re
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

clean_data = pd.read_csv('csv/Tweets.csv')
clean_data.head()

