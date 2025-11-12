"""This script contains code to perform sentiment analysis"""

# %% import libraries
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime
import pandas as pd

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()


# %% Function to get the compound sentiment score using VADER
def get_sentiment(text):
    return sid.polarity_scores(text)["compound"]


# %% __main__
if __name__ == "__main__":
    type = "news"
    timestamp = "2024-11-05_16-19-53"

    # %% Read the svos
    df = pd.read_csv(f"../output/svo_{timestamp}.csv.gz", compression="gzip")

    # %% Apply the function to the 'sentence' column and store the result in a new 'sentiment' column
    df["sentiment"] = df["sentence"].apply(get_sentiment)

    # %% Export the file with the timestamp included
    df.to_csv(
        f"../output/svo_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv.gz",
        compression="gzip",
        index=False,
    )

# %%
