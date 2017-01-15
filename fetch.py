#! /usr/bin/python3
feeds = [
    ('http://feeds.washingtonpost.com/rss/politics', 'wpost'),
    ('http://feeds.feedburner.com/breitbart', 'breitbart'),
    ('http://www.huffingtonpost.com/feeds/verticals/politics/index.xml', 'huffpo'),
#    ('http://rssfeeds.usatoday.com/UsatodaycomWashington-TopStories', 'usatoday'),
    ('http://rss.nbcnews.com/', 'nbcnews'),
    ('http://feeds.abcnews.com/abcnews/politicsheadlines', 'abcnews'),
    ('http://www.wsj.com/xml/rss/3_7085.xml', 'wsj'),
    ('http://feeds.reuters.com/Reuters/PoliticsNews', 'reuters'),
    ('http://www.nytimes.com/services/xml/rss/nyt/Politics.xml', 'nyt'),
    ('http://feeds.foxnews.com/foxnews/politics', 'fox'),
    ('http://rss.cnn.com/rss/cnn_allpolitics.rss', 'cnn')
]
import sys
import feedparser
import pdb
import urllib
from bs4 import BeautifulSoup
from http.cookiejar import CookieJar
import preprocess
import group_articles
import pickle

def fetch_links():
    articles = []
    for feed in feeds:
        url = feed[0]
        feed_id = feed[1]
        d = feedparser.parse(url)
        if not 'entries' in d:
            print('No entries for a parsed feed.')
            continue
        for entry in d.entries:
            link = entry.link
            articles.append(fetch_and_parse(link, feed_id))
    pickle.dump(articles, open("articles.pickle", "wb"))
    group_articles.group(articles)

def fetch_and_parse(link, feed_id):
    print("Fetching link: " + link)
    content = None
    try:
        cj = CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
        request = urllib.request.Request(link)
        request.add_header('User-Agent', 'Mozilla/5.0')
        content = opener.open(request).read()
        #raw_response = response.decode('utf8', errors='ignore')
    except urllib.error.HTTPError as err:
        print('HTTP error: {0}'.format(err))
        return
    except UnicodeEncodeError as err:
        print('UnicodeEncodeError: {0}'.format(err))
        return
    soup = BeautifulSoup(content, "html.parser")
    text = parse_article(soup)
    if text == None:
        return
    title = str(soup.title.get_text().encode('utf-8'))
    print(text)
    print('title: ' + title)
    print('text length: ' + str(len(text)))
    #stems = preprocess.tokenize(text)
    #preprocess.cache_for_analysis(link, title, stems, feed_id)
    # In contrast with the last approach, cache the articles here
    # and pass them to analysis afterwards.
    return (title, text)

# Return the article's text, given a BeautifulSoup object of the page.
def parse_article(soup):
    # kill all script and style elements
    for script in soup(['script', 'style']):
        script.extract()    # rip it out
    # there are actually multiple articles
    articles = soup.find_all('article')
    if len(articles) > 0:
        # use max to get article with max length
        article = max(articles, key=lambda a:len(a.get_text()))
        text = article.get_text().encode('utf-8')
        if len(text) >= 100:
            return text
        else:
            print('Length of text is less than 100. This article seems too short.')
    article_span = soup.find('span', id='article-text')
    if article_span != None:
        text = article_span.get_text().encode('utf-8')
        if len(text) >= 100:
            return text
        else:
            print('Length of text is less than 100. This article seems too short.')
            pdb.set_trace()
    print("Could not find an article in the link.")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        fetch_and_parse(sys.argv[1])
    else:
        fetch_links()

