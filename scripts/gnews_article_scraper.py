"""This script contains code to download news articles from google news"""

# %%
# * Import libraries
import datetime
import pandas as pd
import numpy as np
import ast
from gnews import GNews
from newspaper import Article
import os
import ast
from newspaper import Config


# %%
def check_path_exists(path):
    if os.path.exists(path):
        return True
    else:
        raise FileNotFoundError(f"The specified path '{path}' does not exist.")


# * Define methods
def search_news_articles(
    keyword,
    start_date,
    end_date,
    path=None,
    delta=7,
    verbose=True,
    country="IT",
    language="it",
    # exclude_websites=["yahoo.com"],
):
    """searches for news articles on Google News platform related to the specified keyword from start date until end date.

    Args:
        keyword (_type_): keyword to search for news articles
        start_date (_type_): search for articles published after date in datetime,date(yyyy,mm,dd) format
        end_date (_type_): search for articles published before date in datetime,date(yyyy,mm,dd) format
        delta (int, optional): Interval of days for building search query. Defaults to 7.
        verbose (bool, optional): Print logs. Defaults to True.
        country (str, optional): Country of search. Defaults to "IT".
        language (str, optional): Search for articles published in particular language. Defaults to "it".
        exclude_websites (list, optional): List of websites to exclude. Defaults to ["yahoo.com"].
        path (str, optional): storage location of news articles. Defaults to "../data/".
    """
    file_name = keyword + "-" + str(start_date) + "-to-" + str(end_date) + ".csv"
    delta = datetime.timedelta(days=delta)

    # create an instance of gnews
    google_news = GNews()
    # google_news.period = '7m'  # News from last 7 months
    # google_news.max_results = 10  # number of responses across a keyword
    google_news.country = country  # News from a specific country
    google_news.language = language  # News in a specific language
    # google_news.exclude_websites = exclude_websites

    # iterating through dates and keywords to scape news articles
    news_df = []
    while start_date <= end_date:
        # iterating through the dates and incrementing the dates with delta
        google_news.start_date = start_date
        google_news.end_date = start_date + delta
        news = google_news.get_news(keyword)
        if verbose:
            print(f" Found {len(news)} news articles for {keyword} from {start_date} until {start_date + delta}")

        news_df.append(pd.DataFrame(news))  # create a dataframe of news articles
        start_date = start_date + delta
    news_df = pd.concat(news_df, ignore_index=True)
    if path is not None:
        news_df.to_csv(path + file_name, index=False)
    return news_df


def parse_publisher(d):
    try:
        # Check if d is a string and convert it to a dictionary
        if isinstance(d, str):
            d = ast.literal_eval(d)
        return d["title"]
    except (KeyError, ValueError, SyntaxError):
        return None


def fetch_full_article(url):
    """
    Fetches and retrieves various details from the provided URL.

    Args:
        url (str): The URL of the article to fetch.

    Returns:
      - 'text' (str): The main text content of the article.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# %%
# * main
if __name__ == "__main__":
    # set path
    path = "data/gnews/"
    check_path_exists(path=path)
    # *  Build search query
    # specify start and end dates in yyyy, mm, dd format
    start_date = datetime.date(2023, 4, 1)
    end_date = datetime.date(2024, 3, 1)

    keyword = "alluvione emilia-romagna"

    # %%
    # * Search for articles
    news_df = search_news_articles(keyword=keyword, start_date=start_date, end_date=end_date)
    news_df["publisher"] = news_df["publisher"].apply(parse_publisher)
    # %%
    config = Config()
    config.memoize_articles = False
    config.fetch_images = False
    config.language = "it"

    # * Download full articles
    news_df.loc[:, "article"] = news_df["url"].apply(fetch_full_article)

    # * Format output
    print(f"Total articles on Google News Archive for keyword {keyword} = {len(news_df)}")
    print(f"Unable to retrieve full text for {len(news_df[news_df['text'].isna()])} articles")
    print(f"Full text downloaded for {len(news_df[news_df['text'].notna()])} articles")

    # Remove articles entries for which full text are not available. Total 18 such entries are removed from the dataframe.
    news_df.dropna(subset="text", inplace=True)
    news_df.reset_index(drop=True, inplace=True)

    # %%
    # * Store output
    news_df.to_csv(
        f"{path}{keyword}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv.gz",
        index=False,
        compression="gzip",
    )
