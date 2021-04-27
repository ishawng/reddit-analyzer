import pickle
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sp

data = pd.read_pickle('corpus.pkl')

# Perform sentiment analysis
data['polarity'] = data['post'].apply(lambda l: TextBlob(l).sentiment.polarity)
data['subjectivity'] = data['post'].apply(lambda l: TextBlob(l).sentiment.subjectivity)
pd.set_option('display.max_columns', None)

plt.rcParams['figure.figsize'] = [10, 8]
plt.xlim(-1.1, 1.1)
plt.ylim(-0.1, 1.1)
x_arr = []
y_arr = []
for i, post in enumerate(data.index):
    x = data.polarity.loc[post]
    y = data.subjectivity.loc[post]
    if x == 0 and y == 0:
        continue
    x_arr.append(x)
    y_arr.append(y)

    plt.scatter(x, y, color='blue')
    # if x <= -0.25 or x >= 0.75 or y <= 0.2 or y >= 0.8:
    #     plt.text(x + 0.001, y + 0.001, data.index[i], fontsize=10)

# plt.hist2d(x_arr, y_arr, bins=[np.arange(-1, 1, 0.05), np.arange(0, 1, 0.025)])
plt.title('Sentiment Analysis', fontsize=20)
plt.xlabel('<-- Negative ------- Positive -->', fontsize=15)
plt.ylabel('<-- Facts ------- Opinions -->', fontsize=15)

plt.show()

data_combined = []
for post in data.index:
    data_combined.append(data['post'][post])
subreddit_text = ' '.join(data_combined)
subreddit_polarity = TextBlob(subreddit_text).sentiment.polarity
subreddit_subjectivity = TextBlob(subreddit_text).sentiment.subjectivity

print(f'Subreddit Polarity and Subjectivity: {subreddit_polarity}, {subreddit_subjectivity}')

