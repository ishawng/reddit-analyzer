# Web scraping and pickle imports
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import pickle
import os
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer


# Scrape post data from link to post on reddit.com
def url_to_post(url):
    try:
        old_reddit_url = url.replace('www', 'old', 1)
        ua = UserAgent()
        page = requests.get(old_reddit_url, headers={'User-Agent': ua.random}).text

        soup = BeautifulSoup(page, 'html.parser')
        post_header = soup.find(class_='top-matter').find('a').text
        post_body = [p.text for p in soup.find(class_='expando').find_all('p')]
        comments = [div.find(class_='md').find('p').text for div in soup.find(class_='commentarea')
                    .find_all('div', class_='entry')]

        post_content = [post_header] + post_body + comments
        return post_content
    except AttributeError:
        return []


# Get urls to all posts on first page of subreddit given subreddit link
def subreddit_to_urls(subreddit_url, pages):
    old_reddit_url = subreddit_url.replace('www', 'old', 1)
    ua = UserAgent()
    page = requests.get(old_reddit_url, headers={'User-Agent': ua.random}).text

    soup = BeautifulSoup(page, 'html.parser')

    # Helper method to find elements that are valid posts
    def is_valid_post(element):
        return element.has_attr('class') and 'thing' in element['class'] and not any(c in ['stickied', 'promotedlink']
                                                                                     for c in element['class'])
    all_posts = []
    for _ in range(pages):
        posts = ['https://www.reddit.com' + div['data-permalink'] for div in soup.find(id='siteTable')
                 .find_all(is_valid_post)]
        all_posts += posts
        next_url = soup.find(class_='next-button')
        if next_url is None:
            break
        next_url = next_url.find('a')['href']
        page = requests.get(next_url, headers={'User-Agent': ua.random}).text
        soup = BeautifulSoup(page, 'html.parser')
    return all_posts


# Convert post URL to post ID (redd.it/postID)
def url_to_id(url):
    return url[url.index('comments/') + 9:url.index('comments/') + 15]


# Convert list of text into string (key: postID, value: string format)
def combine_text(list_of_text):
    return ' '.join(list_of_text)


sub = 'https://www.reddit.com/r/gatech/'
num_pages = 20
urls = subreddit_to_urls(sub, num_pages)
# Pickle posts for later use
for post_url in urls:
    content = url_to_post(post_url)
    if len(content) > 1:
        with open('posts/' + url_to_id(post_url) + '.txt', 'wb') as file:
            pickle.dump(content, file)

# Load pickled files
data = {}
for _, file_name in enumerate(os.listdir('posts')):
    with open('posts/' + file_name, 'rb') as file:
        data[file_name[:-4]] = pickle.load(file)

data_combined = {key: [combine_text(value)] for (key, value) in data.items()}

# Put data into pandas dataframe
pd.set_option('max_colwidth', 150)

# Generate corpus
data_df = pd.DataFrame.from_dict(data_combined).transpose()
data_df.columns = ['post']
data_df = data_df.sort_index()
data_df.to_pickle('corpus.pkl')


# Clean the data
def clean_text(text):
    text = text.lower()
    text = re.sub('/', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub('[^A-Za-z0-9 ]', '', text)
    text = re.sub('\n', '', text)
    return text


data_clean = pd.DataFrame(data_df.post.apply(lambda x: clean_text(x)))
data_clean.to_pickle('data_clean.pkl')

# Generate document term matrix
cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.post)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index
data_dtm.to_pickle('dtm.pkl')
pickle.dump(cv, open('cv.pkl', 'wb'))
