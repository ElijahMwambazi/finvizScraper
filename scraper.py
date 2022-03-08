from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from datetime import date
import numpy as np
import pandas as pd
import requests
import os
import re

finviz_url = "https://finviz.com/quote.ashx?t="
tickers = ["AMZN", "AAPL", "GOOGL", "MSFT", "TSLA", "JPM", "JNJ", "FB", "V"]
today = date.today()
news_tables = {}

# Get Company Equivalent Of Ticker
def get_symbol(symbol):
    try:
        symbol_list = requests.get(
            f"http://chstocksearch.herokuapp.com/api/{symbol}"
        ).json()
    except ():
        print("Check internet connection.")

    for x in symbol_list:
        if x["symbol"] == symbol:
            return x["company"]


for ticker in tickers:
    url = finviz_url + ticker
    # company = get_symbol(ticker)

    # Stock And Forex News Data Attained From "https://Finviz.com/
    req = Request(url=url, headers={"user-agent": "my-app"})
    response = urlopen(req)

    # Parsing HTML Code For New Data
    html = BeautifulSoup(response, "html.parser")
    news_table = html.find(id="news-table")
    news_tables[ticker] = news_table

parsed_data = []

for ticker, news_table in news_tables.items():

    for row in news_table.findAll("tr"):

        title = row.a.get_text()
        date_data = row.td.text.split(" ")

        # Check Whether date only or time inclusive
        if len(date_data) == 1:
            time = date_data[0]
        else:
            Date = date_data[0]
            time = date_data[1]

        # Add ticker, date, time, title To parsed_data List And Pass It To A Dataframe df
        parsed_data.append([ticker, Date, time, title])

df = pd.DataFrame(parsed_data, columns=["ticker", "date", "time", "title"])

# Create Sentiment Analysis Analyzer
sen_analysis = SentimentIntensityAnalyzer()

score_calculator_neg = lambda title: sen_analysis.polarity_scores(title)["neg"]
df["negative score"] = df["title"].apply(score_calculator_neg)

score_calculator_neu = lambda title: sen_analysis.polarity_scores(title)["neu"]
df["neutral score"] = df["title"].apply(score_calculator_neu)

score_calculator_pos = lambda title: sen_analysis.polarity_scores(title)["pos"]
df["positive score"] = df["title"].apply(score_calculator_pos)

score_calculator_compound = lambda title: sen_analysis.polarity_scores(title)[
    "compound"
]
df["compound score"] = df["title"].apply(score_calculator_compound)

# Removing Unwanted Charaters from Time Column
for i in df["time"].values:
    df["time"] = re.sub("Â Â ", "", str(i))

# Convert String Date to Datetime Variables
df["date"] = pd.to_datetime(df.date).dt.date

# Add Label Column To DataFrame
df["label"] = np.where(df["compound score"] < 0, "negative", "positive")
df["label"] = np.where(df["compound score"] == 0, "neutral", df["label"])

# Group DataFrames by their tickers
grouped_df = df.groupby(df.ticker)

# Creating directory
directory = "scraped Data"
path_dir = os.getcwd()
if not os.path.exists(directory):
    os.mkdir(os.path.join(path_dir, directory))
else:
    pass

# Save DataFrame As CSV File
for ticker in tickers:
    # ticker_names = "_".join(tickers)
    try:
        grouped_df.get_group(ticker).to_csv(
            os.path.join(directory, f"{today} sentiment data {ticker}.csv"),
            index=False,
            header=True,
        )
    except Exception as e:
        print(f"{e} File in use.")

print("DONE..")
