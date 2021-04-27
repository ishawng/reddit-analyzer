import pandas as pd
from collections import Counter
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random

# Read in the document term matrix
data = pd.read_pickle('dtm.pkl')
data = data.transpose()

top_words_dict = {}
for post in data.columns:
    top = data[post].sort_values(ascending=False).head(30)
    top_words_dict[post] = list(zip(top.index, top.values))

# for post, top_words in top_words_dict.items():
#     print(post)
#     print(', '.join([word for word, count in top_words[:15]]))
#     print('----')

words = []
for post in data.columns:
    top = [word for (word, count) in top_words_dict[post]]
    for t in top:
        words.append(t)

additional_stop_words = [word for word, count in Counter(words).most_common() if count >= 7]

# Update document term matrix with additional stop words
data_clean = pd.read_pickle('data_clean.pkl')
stop_words = text.ENGLISH_STOP_WORDS.union(additional_stop_words)
cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean.post)
data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_stop.index = data_clean.index
pickle.dump(cv, open('cv_stop.pkl', 'wb'))
data_stop.to_pickle('dtm_stop.pkl')

# Generate word clouds for visual analysis
wc = WordCloud(stopwords=stop_words, background_color='white', colormap='Dark2', max_font_size=150, random_state=12)
plt.rcParams['figure.figsize'] = [16, 6]
post_ids = [pid for pid in data_clean.index]

index = 0
for _ in range(12):
    i = random.randint(0, len(data.columns))
    wc.generate(data_clean.post[i])

    plt.subplot(3, 4, index + 1)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(post_ids[i])
    index += 1

plt.show()

# Find most common words in entire subreddit
subreddit_top = pd.DataFrame()
subreddit_top['Sum'] = data_stop.transpose().sum(axis=1)
data_combined = subreddit_top.sort_values(by='Sum', ascending=False).head(30)
top_subreddit_words = {}
for (word, count) in data_combined['Sum'].items():
    top_subreddit_words[word] = count
# print(top_subreddit_words)

