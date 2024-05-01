# Databricks notebook source
pip install yfinance

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

#import yfinance as yf
import yfinance as yf
import pandas as pd
import os

# COMMAND ----------

# Define the ticker symbol and period for which we want to fetch the stock prices
ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2024-04-30"

# Fetch the stock prices for the specified period
data = yf.download(ticker, start=start_date, end=end_date)

# Print the fetched stock prices
print(data)

# COMMAND ----------

df = pd.DataFrame(data)

# COMMAND ----------

df.head()

# COMMAND ----------

df.info

# COMMAND ----------

df['date'] = pd.to_datetime(df.index)

# COMMAND ----------

df.head()

# COMMAND ----------

import plotly.graph_objects as go

# Create the cadlestick chart
fig = go.Figure(data=[go.Candlestick(x=df['date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

# Customize the chart Layout
fig.update_layout(
    title='Stock Price Chart AAPL',
    yaxis_title='Price ($)',
    xaxis_rangeslider_visible=False)

# Display the chart
fig.show()

# COMMAND ----------

df.drop(['date', 'Volume'], axis=1, inplace=True)

# COMMAND ----------

df.reset_index(drop=True, inplace=True)

# COMMAND ----------

df.plot.line(y="Close", use_index=True)

# COMMAND ----------

df.reset_index(drop=True, inplace=True)

# COMMAND ----------

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# # Load the dataset
# df = pd.csv('stock_price.csv)

# Split the dataset into training and testing sets
x = df[['Open', 'Close', 'High', 'Low', 'Adj Close']] # Input features
y = df['Close'] # Target variable
X_train, X_test, y_tarin, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Build the Random Forest Regression model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_tarin)

# Make predictions on the testing set
y_pred = rf.predict(X_test)

# Evaluate the model using mean sqared error
mse = mean_squared_error(y_test, y_pred)
print('Mean Sqared Error:', mse)

# COMMAND ----------

import numpy as np

# Input values to predict the stock price
new_data = np.array([[169.529999, 170.610001, 168.149994, 169.889999, 169.889999]])

# Make predictions using the trained model
predicated_price = rf.predict(new_data)

# Print the predicated stock price
print('Predicated Stock Price:', predicated_price[0])

# COMMAND ----------

df.tail()
