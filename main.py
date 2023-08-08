from flask import Flask, request, jsonify, render_template, Blueprint
import pandas as pd
import json
# Import libraries
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import plotly.express as px
# NLTK VADER for sentiment analysis
import nltk
import requests
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import pandas as pd
import hvplot.pandas
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pickle

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

file_path = "Resources/all_stock_complete.csv"
df_all_stock= pd.read_csv(file_path)


loaded_model = joblib.load('rf_model.pkl')

# # # for extracting data from finviz
# # #finviz_url = 'https://finviz.com/quote.ashx?t='

app = Flask(__name__)

#  Define a function to get data for the selected ticker
def get_ticker_data(ticker):
    
    ticker_data = df_all_stock[df_all_stock['ticker'] == ticker]
    
    if not ticker_data.empty:
        # Get the last quarter's Sharpe Ratio for the selected ticker
        sharpe_ratio = ticker_data.iloc[-1]['sharpe_ratio']

        return sharpe_ratio

    return None

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        ticker = request.form['ticker']

        sharpe_ratio = get_ticker_data(ticker)
        # Get the data for the selected ticker
        # ticker_data = get_ticker_data(ticker)

        if sharpe_ratio is not None:

            price = float(request.form['price'])
            volume = float(request.form['volume'])
            

            prediction = loaded_model.predict([[price, volume, sharpe_ratio]])[0]

            # Get the response from the '/api/v1.0/parse_and_score_news' endpoint with the selected ticker
        response = requests.get(f'http://127.0.0.1:5000/api/v1.0/parse_and_score_news?ticker={ticker}')
        parsed_and_scored_news = pd.read_json(response.text)

        # Filter the parsed_and_scored_news DataFrame for the specified ticker
        ticker_data = parsed_and_scored_news[parsed_and_scored_news['ticker'] == ticker]

        # Group by date and ticker columns from ticker_data and calculate the mean
        mean_scores = ticker_data.resample('W', on='datetime')['sentiment_score'].mean()

        # Convert the result to a DataFrame
        mean_scores = mean_scores.reset_index()
        mean_scores.columns = ['datetime', 'sentiment_score']  # Set the datetime column name

        # Plot a bar chart with plotly
        fig = px.bar(mean_scores, x='datetime', y='sentiment_score', title=ticker + ' Weekly Sentiment Scores')

        # Calculate the mean sentiment score for the selected ticker
        overall_average_score = ticker_data['sentiment_score'].mean()

        # Convert the overall_average_score to a string with 2 decimal places
        overall_average_score_str = "{:.2f}".format(overall_average_score)

        # Convert the figure to HTML
        plot_html = fig.to_html()

        submitted = True  # The form has been submitted

        return render_template('result3.html', prediction=prediction, ticker=ticker, sharpe_ratio=sharpe_ratio, price=price, volume=volume, submitted=submitted, overall_average_score=overall_average_score_str, plot_html=plot_html)
    
    return render_template('index3.html')

@app.route('/get_sharpe_ratio', methods=['GET'])
def get_sharpe_ratio():
    ticker = request.args.get('ticker')

    # Get the last quarter's Sharpe Ratio for the selected ticker
    sharpe_ratio = get_ticker_data(ticker)

    if sharpe_ratio is not None:
        return jsonify({'sharpe_ratio': sharpe_ratio})
    else:
        return jsonify({'sharpe_ratio': 'N/A'})

def get_news():
    tickers = ['HLT', 'MAR', 'CCL', 'WMT', 'WH', 'AMZN', 'UAL', 'DAL', 'CAKE', 'RCL']
    finviz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}  # Initialize an empty dictionary to store news tables

    for ticker in tickers:
        url = finviz_url + ticker
        req = Request(url=url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'})
        response = urlopen(req)
        html = BeautifulSoup(response, features="html.parser")
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table

    return news_tables

@app.route('/api/v1.0/get_news', methods=['GET'])
def get_news_endpoint():
    news_tables = get_news()

    # Convert the BeautifulSoup Tag objects to strings before returning the response
    news_tables_str = {ticker: str(table) for ticker, table in news_tables.items()}
    
    return jsonify(news_tables_str)


@app.route('/api/v1.0/parse_and_score_news', methods=['GET'])
def parse_and_score_news():

    news_tables = get_news()  # Get news_tables from the get_news function
    parsed_news = []

    for ticker, news_table in news_tables.items():
        for x in news_table.findAll('tr'):
            a_element = x.a

            if a_element is not None:
                text = a_element.get_text()

                td_text = x.td.text.split()
                if len(td_text) == 1:
                    time = td_text[0]
                    date = None
                else:
                    date = td_text[0]
                    time = td_text[1]

                # Append ticker, date, time, and headline as a dictionary to the 'parsed_news' list
                parsed_news.append({'ticker': ticker, 'date': date, 'time': time, 'headline': text})

    # Convert the parsed_news list into a DataFrame called 'parsed_news_df'
    parsed_news_df = pd.DataFrame(parsed_news)
    parsed_news_df['datetime'] = pd.to_datetime(parsed_news_df['date'] + ' ' + parsed_news_df['time'], format="%b-%d-%y %I:%M%p")
    parsed_news_df.drop(columns=['date', 'time'], inplace=True)
    parsed_news_df['datetime'] = parsed_news_df['datetime'].dt.strftime('%m/%d/%Y %H:%M')

    # Instantiate the sentiment intensity analyzer
    vader = SentimentIntensityAnalyzer()

    # Iterate through the headlines and get the polarity scores using vader
    scores = parsed_news_df['headline'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)

    # Join the DataFrames of the news and the list of dicts
    parsed_and_scored_news = parsed_news_df.join(scores_df, rsuffix='_right')
    parsed_and_scored_news = parsed_and_scored_news.set_index('datetime')
    parsed_and_scored_news = parsed_and_scored_news.drop(['headline'], axis=1)
    parsed_and_scored_news = parsed_and_scored_news.rename(columns={"compound": "sentiment_score"})

    # Convert the final DataFrame to JSON response
    json_response = parsed_and_scored_news.reset_index().to_json(orient='records')

    # Create a Flask response with the JSON data and set the content type
    response = app.response_class(response=json_response, status=200, mimetype='application/json')

    return response


@app.route('/api/v1.0/plot_hourly_sentiments', methods=['GET'])
def plot_hourly_sentiments():
    # Get the response from the '/api/v1.0/parse_and_score_news' endpoint
    #response = requests.get('http://127.0.0.1:5000/api/v1.0/parse_and_score_news')
    #parsed_and_scored_news = pd.read_json(response.text)

    response = requests.get('http://127.0.0.1:5000/api/v1.0/parse_and_score_news')
    json_response = response.json()  # Parse the JSON data

    parsed_and_scored_news = pd.DataFrame.from_records(json_response)



    # Example list of tickers
    tickers = ['HLT', 'MAR', 'CCL', 'WMT', 'WH', 'AMZN', 'UAL', 'DAL', 'CAKE','RCL']
    # List to store the plots for each ticker
    plots = []

    for ticker in tickers:
        # Filter the parsed_and_scored_news DataFrame for the current ticker
        ticker_data = parsed_and_scored_news[parsed_and_scored_news['ticker'] == ticker]
        
        # Group by date and ticker columns from ticker_data and calculate the mean
        mean_scores = ticker_data.resample('W', on='datetime')['sentiment_score'].mean()

        # Convert the result to a DataFrame
        mean_scores = mean_scores.reset_index()
        mean_scores.columns = ['datetime', 'sentiment_score']  # Set the datetime column name

        # Plot a bar chart with plotly
        fig = px.bar(mean_scores, x='datetime', y='sentiment_score', title=ticker + ' Weekly Sentiment Scores')
        
        plots.append(fig)
    

    return jsonify(plots)


@app.route("/", methods=['GET', 'POST'])
def welcome():
    submitted = False  # Flag to check if the form has been submitted
    plot_html = None  # Variable to store the plot HTML
    ticker = None  # Initialize ticker with None
    overall_average_score_str = None  # Initialize overall_average_score_str with None


if __name__ == '__main__':
    app.run(debug=True)
