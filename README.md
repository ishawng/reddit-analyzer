# Subreddit Analyzer
## Introduction
We start by taking a URL to a subreddit on [Reddit](https://www.reddit.com) and scraping recent posts from the subreddit. We then clean the text contained in these posts and their relevant comments. Then, we perform some initial exploratory data analysis on the text. We find the top words used in each post, the top words used in the subreddit as a whole, and we visualize the most used words in 12 random posts using [word clouds](https://en.wikipedia.org/wiki/Tag_cloud). After this initial analysis, we apply natural language processing methods such as sentiment analysis and topic modelling. We perform sentiment analysis for each individual post, as well as the subreddit as a whole. Then, we generate topics that describe possible groupings of posts.

## Data Collection
The following steps are performed in `reddit_data_collection.py`. We start by scraping approximately 500 of the most recent posts from the given subreddit using [requests](https://docs.python-requests.org/en/master/) and [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/). Then, we save the contents of these posts using [pickle](https://docs.python.org/3/library/pickle.html). We insert this text data into a [pandas](https://pandas.pydata.org/) dataframe object and create our corpus. We also save this dataframe using pickle. Afterwards, we clean the text data using various methods. We load the cleaned data into another dataframe and save it using pickle. Finally, we use a [sklearn](https://scikit-learn.org/stable/) CountVectorizer to generate a document-term matrix. We save both the document-term matrix and the CountVectorizer object using pickle.

## Exploratary Data Analysis
The following steps are performed in `reddit_exploratary_data_analysis.py`. We start by loading the previously saved document-term matrix from a file. We convert this into a term-document matrix to more easily perform analysis. We find the most frequently used words in each post by sorting the term-document matrix by its columns. Then, we load the previously saved cleaned data from a file. We add additional stop words using some of the most frequently found words in each post. Then, we save text data into a dataframe and save it using pickle. Afterwards, we generate word clouds to visualize the most used words in 12 random posts. These are generated using [WordCloud](http://amueller.github.io/word_cloud/). These clouds are drawn using [matplotlib](https://matplotlib.org/). Some examples are displayed below, along with their post IDs. Then, we aggregate the text data and find the most used words in the subreddit as a whole.
![Word clouds](https://raw.githubusercontent.com/ishawng/subreddit-analyzer/main/wordclouds.png)

## Sentiment Analysis
The following steps are performed in `reddit_sentiment_analysis.py`. We begin by loading the previously saved corpus from a file. Then, we perform sentiment analysis using [TextBlob](https://textblob.readthedocs.io/en/dev/). Using this data and matplotlib, we plot each post on a scatterplot (subjectivity vs. polarity). We also generate a heatmap of the subreddit using this data. The scatterplot and heatmap for the [Georgia Tech subreddit](https://www.reddit.com/r/gatech) are shown below. Finally, by using the aggregate text data, we find the polarity and subjectivity for the subreddit as a whole. In the Georgia Tech subreddit's case, the polarity and subjectivity were approximately 0.130 and 0.493 respectively.
![Scatterplot](https://raw.githubusercontent.com/ishawng/subreddit-analyzer/main/scatterplot.png)
![Heatmap](https://raw.githubusercontent.com/ishawng/subreddit-analyzer/main/heatmap.png)

## Topic Modelling
The following steps are performed in `reddit_topic_modelling.py`. We begin by loading the previously saved document-term matrix which was filtered for stop words. We again convert this into a term-document matrix for easier analysis. Then, we convert this into a sparse matrix and generate a corpus using [scipy](https://www.scipy.org/) and [NLTK](https://www.nltk.org/). We also filter out words that are not nouns or adjectives. This allows us to obtain more reasonable topics. Then, using Latent Dirichlet Allocation with [gensim](https://radimrehurek.com/gensim/), we generate possible topics for the posts. Finally, we consider the options that are generated and select those which seem reasonable considering the context. In the Georgia Tech subreddit's case, these topics were "campus", "vaccine", "police", "students", and "mental [health]".

## Further Improvements
The process of select valid posts can be improved in various ways. For example, we can choose to ignore posts that contain fewer than a certain number of words. We can also choose to ignore posts with only images and no body text. Also, we can add more specific stop words in the context of each subreddit, although this would likely have to be done manually.
